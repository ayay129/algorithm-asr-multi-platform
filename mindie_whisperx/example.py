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

import argparse
import time
import librosa
from pipeline.pipeline import MindiePipeline, load_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-whisper_model_path', type=str, required=True, help="please provide model path.")
    parser.add_argument('-vad_model_path', type=str, required=True)
    parser.add_argument('-audio_path', type=str, required=True)
    parser.add_argument('-bs', type=int, required=True, help="please provide batch_size.")
    parser.add_argument('-compiled_models', type=str, required=True, help="compiled models save dir.")
    parser.add_argument('-device_id', type=int, default=0, help="please provide device id")
    parser.add_argument('-open_warm_up', type=bool, default=False, help="open warm up or not")
    args = parser.parse_args()
    mindie_pipe = MindiePipeline(
        args.whisper_model_path,
        args.vad_model_path,
        args.compiled_models,
        args.bs,
        args.device_id
    )
    y, audio_sr = librosa.load(args.audio_path)
    duration_seconds = librosa.get_duration(y=y, sr=audio_sr)
    print(f"duration_seconds {duration_seconds}")
    speech_data = load_audio(args.audio_path)
    if args.open_warm_up:
        predicted_ids = mindie_pipe.transcribe(speech_data, batch_size=args.bs)
        print(f"warm up success.")
    t0 = time.time()
    predicted_ids = mindie_pipe.transcribe(speech_data, batch_size=args.bs)
    print(f"trascription {predicted_ids}")
    t1 = time.time()
    print(f"QPS {duration_seconds / (t1 - t0)}, speech durarations {duration_seconds}")