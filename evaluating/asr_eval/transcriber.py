from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from .config import ModelSpec


class WhisperTranscriber:
    """Wrap model, tokenizer and processor so workers can stay small."""

    def __init__(self, model_spec: ModelSpec, device_id: int, whisper_language: str) -> None:
        self.model_spec = model_spec
        self.whisper_language = whisper_language
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        self.model = WhisperForConditionalGeneration.from_pretrained(model_spec.model_path).to(self.device)
        self.model.eval()

        tokenizer = WhisperTokenizer.from_pretrained(
            model_spec.model_path,
            language=whisper_language,
            task="transcribe",
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_spec.model_path)
        self.processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.tokenizer = tokenizer

        try:
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=whisper_language,
                task="transcribe",
            )
        except Exception:
            # 部分自定义 tokenizer 不支持该接口，此时退回模型默认生成配置。
            self.forced_decoder_ids = None

    def transcribe_array(self, audio_array: Any, sampling_rate: int) -> str:
        inputs = self.processor(
            audio_array,
            return_tensors="pt",
            sampling_rate=sampling_rate,
        )
        input_features = inputs["input_features"].to(self.device)

        generation_kwargs = dict(self.model_spec.generation_kwargs)
        if self.forced_decoder_ids is not None and "forced_decoder_ids" not in generation_kwargs:
            generation_kwargs["forced_decoder_ids"] = self.forced_decoder_ids

        with torch.no_grad():
            generated_ids = self.model.generate(input_features, **generation_kwargs)

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def close(self) -> None:
        # 评测脚本通常会频繁切模型，显式释放显存更稳妥。
        del self.model
        del self.processor
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

