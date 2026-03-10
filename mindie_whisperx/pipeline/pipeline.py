# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from functools import lru_cache
from typing import Union, Optional
import time
import argparse

import torch
import torch.nn.functional as F
import mindietorch
import librosa
import tokenizers
import numpy as np
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator
from transformers import WhisperProcessor
from whisper_large_v3.modeling_whisper import MindieWhisperForConditionalGeneration
from .vad import load_vad_model, merge_chunks
from .tokenizer import Tokenizer


def exact_div(x1, x2):
    if x1 % x2 != 0:
        raise ValueError("x1 is not divisible by x2")
    return x1 // x2


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    mel_filters_path = os.path.join(os.path.dirname(__file__), "mel_filters.npz")
    if not os.path.exists(mel_filters_path):
        np.savez_compressed(
            mel_filters_path,
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    if n_mels not in [80, 128]:
        raise ValueError(f"Unsupported n_mels: {n_mels}")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters("cpu", n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if log_spec.shape[-1] != 3000:
        raise ValueError("log_spec shape is not as expected")
    return log_spec


class MindiePipeline(Pipeline):
    def __init__(self, whisper_model_path, vad_model_path, save_path, batch_size, deivce_id, **kwargs):
        self.model = MindieWhisperForConditionalGeneration.from_pretrained(whisper_model_path).to("cpu")
        # 32g
        os.environ["TORCH_AIE_NPU_CACHE_MAX_SIZE"] = "32"
        self.device = torch.device("cpu")
        if not isinstance(self.model, MindieWhisperForConditionalGeneration):
            raise ValueError(f"Please provide MindieWhisperForConditionalGeneration, found {type(self.model)}")

        if not (save_path and batch_size):
            raise ValueError(f"Please provide compiled model save path and batch_size.")

        mindietorch.set_device(deivce_id)
        self.model.load_mindie_models(save_path, batch_size)
        print("start load vad")

        tokenizer_path = os.path.join(whisper_model_path, "tokenizer.json")
        self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        self.tokenizer = Tokenizer(self.hf_tokenizer, multilingual=True, task="transcribe", language="zh")
        self.default_language = "zh"

        default_vad_options = {
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }

        vad_ts_model_path = os.path.join(save_path, "mindie_vad.ts")
        if not os.path.exists(vad_ts_model_path):
            raise ValueError(f"Expect file name is {vad_ts_model_path}, but can`t be found in path: {save_path}")

        self.vad_model = load_vad_model(vad_model_path, torch.device("cpu"), vad_ts_model_path, **default_vad_options)
        print("load vad success")

        self._batch_size = batch_size
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = "pt"

        super(Pipeline, self).__init__()

        self._vad_params = {
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }
        self.vad_cost = 0


    def _sanitize_parameters(self, **kwargs):
        forward_params = {}
        if "forced_decoder_ids" in kwargs and kwargs["forced_decoder_ids"] is not None:
            forward_params["forced_decoder_ids"] = kwargs["forced_decoder_ids"]
        return {}, forward_params, {}

    def _build_forced_decoder_ids(self, language: str):
        language_code = language.lower()
        language_token = self.hf_tokenizer.token_to_id(f"<|{language_code}|>")
        if language_token is None:
            raise ValueError(
                f"Unsupported language '{language}'. Please pass a Whisper language code such as zh/en/ja/de/fr."
            )
        return [
            [1, language_token],
            [2, self.tokenizer.transcribe],
        ]

    def preprocess(self, audio):
        model_n_mels = 128
        audio = audio['inputs']
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - len(audio),
            device="cpu"
        )
        return {'inputs': features}

    def _forward(self, model_inputs, **generate_kwargs):
        generate_kwargs["input_features"] = model_inputs["inputs"]
        tokens = self.model.generate(attention_mask=None, **generate_kwargs)
        tokens_batch = [x for x in tokens]

        def decode_batch(tokens) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < self.tokenizer.eot])
            return self.tokenizer.tokenizer.decode_batch(res)
        
        text = decode_batch(tokens_batch)
        return {'text': text}
    
    def forward(self, model_inputs, **forward_params):
        model_outputs = self._forward(model_inputs, **forward_params)
        return model_outputs

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        
        return final_iterator
    
    def pad_segment(self, vad_segments, batch_size):
        need_pad_num = batch_size - len(vad_segments) % batch_size
        paded_segments = vad_segments
        for _ in range(need_pad_num):
            paded_segments.append(vad_segments[-1])
        return paded_segments

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, chunk_size=30
    ):
        segments = []
        if batch_size in [0, None]:
            batch_size = self._batch_size

        def data(total_audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                yield {'inputs': total_audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params.get("vad_onset", 0.500),
            offset=self._vad_params.get("vad_offset", 0.363),
        )
        total_segments = len(vad_segments)

        if total_segments % batch_size != 0:
            vad_segments = self.pad_segment(vad_segments, batch_size)

        infer_kwargs = {}
        if language:
            infer_kwargs["forced_decoder_ids"] = self._build_forced_decoder_ids(language)

        for idx, out in enumerate(
            self.__call__(
                data(audio, vad_segments),
                batch_size=batch_size,
                num_workers=num_workers,
                **infer_kwargs,
            )
        ):
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        return segments
    

if __name__ == "__main__":
    print("start here")

    parser = argparse.ArgumentParser()
    parser.add_argument('-whisper_model_path', type=str, required=True, help="please provide model path.")
    parser.add_argument('-vad_model_path', type=str, required=True)
    parser.add_argument('-machine_type', type=str, required=True, choices=["300IPro", "800IA2"])
    parser.add_argument('-audio_path', type=str, required=True)
    parser.add_argument('-bs', type=int, default=16, help="please provide batch_size, default:8.")
    parser.add_argument('-save_path', type=str, default="compiled_models", help="compiled models save dir.")
    parser.add_argument('-device_id', type=int, default=0)

    args = parser.parse_args()
    if args.machine_type == "800IA2":
        from modeling_whisper_800IA2 import MindieWhisperForConditionalGeneration
    elif args.machine_type == "300IPro":
        from modeling_whisper_300IPro import MindieWhisperForConditionalGeneration
    else:
        raise ValueError("machine type is not supported.")
    inference_device = f"npu:{args.device_id}"

    mindie_pipe = MindiePipeline(args.whisper_model_path, args.vad_model_path, inference_device, args.save_path, args.bs)

    mindietorch.set_device(args.device_id)

    audio_path = args.audio_path
    inp = []

    infer_audio = load_audio(audio_path)
    print(f"load audio success.")

    y, audio_sr = librosa.load(audio_path)
    duration_seconds = librosa.get_duration(y=y, sr=audio_sr)
    print(f"duration_seconds {duration_seconds}")

    inp.extend(infer_audio)
    inp = np.array(inp)

    predicted_ids = mindie_pipe.transcribe(inp, batch_size=args.bs)

    t0 = time.time()
    predicted_ids = mindie_pipe.transcribe(inp, batch_size=args.bs)
    print(f"trascription {predicted_ids}")
    t1 = time.time()
    
    print(f"speech_duration/s: {duration_seconds}")
    print(f"E2E cost {t1 - t0}")
    print(f"perfomence {duration_seconds / (t1 - t0)}")
