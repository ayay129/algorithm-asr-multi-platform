from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence


_WHITESPACE_RE = re.compile(r"\s+")
_CJK_LANGUAGES = {"zh", "ja", "ko", "yue"}


def normalize_text(text: str) -> str:
    """Keep normalization conservative so the metric stays explainable."""

    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.casefold()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _tokenize_for_wer(text: str, language: str) -> List[str]:
    if language.lower() in _CJK_LANGUAGES:
        return [character for character in text if not character.isspace()]
    return text.split()


def _tokenize_for_cer(text: str) -> List[str]:
    return [character for character in text if not character.isspace()]


def _levenshtein_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    previous_row = list(range(len(prediction) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current_row = [ref_index]
        for pred_index, pred_token in enumerate(prediction, start=1):
            insert_cost = current_row[pred_index - 1] + 1
            delete_cost = previous_row[pred_index] + 1
            replace_cost = previous_row[pred_index - 1] + (ref_token != pred_token)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


@dataclass
class SampleScore:
    normalized_reference: str
    normalized_prediction: str
    wer: float
    cer: float


@dataclass
class MetricAccumulator:
    """Accumulate corpus-level metrics instead of averaging sample-level scores."""

    sample_count: int = 0
    failed_count: int = 0
    word_errors: int = 0
    word_reference_tokens: int = 0
    char_errors: int = 0
    char_reference_tokens: int = 0

    def add(self, reference_text: str, prediction_text: str, language: str) -> SampleScore:
        normalized_reference = normalize_text(reference_text)
        normalized_prediction = normalize_text(prediction_text)

        reference_words = _tokenize_for_wer(normalized_reference, language)
        prediction_words = _tokenize_for_wer(normalized_prediction, language)
        word_errors = _levenshtein_distance(reference_words, prediction_words)

        reference_chars = _tokenize_for_cer(normalized_reference)
        prediction_chars = _tokenize_for_cer(normalized_prediction)
        char_errors = _levenshtein_distance(reference_chars, prediction_chars)

        self.sample_count += 1
        self.word_errors += word_errors
        self.word_reference_tokens += max(1, len(reference_words))
        self.char_errors += char_errors
        self.char_reference_tokens += max(1, len(reference_chars))

        return SampleScore(
            normalized_reference=normalized_reference,
            normalized_prediction=normalized_prediction,
            wer=word_errors / max(1, len(reference_words)),
            cer=char_errors / max(1, len(reference_chars)),
        )

    def add_failure(self) -> None:
        self.failed_count += 1

    def as_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "failed_count": self.failed_count,
            "wer": self.word_errors / max(1, self.word_reference_tokens),
            "cer": self.char_errors / max(1, self.char_reference_tokens),
            "word_errors": self.word_errors,
            "word_reference_tokens": self.word_reference_tokens,
            "char_errors": self.char_errors,
            "char_reference_tokens": self.char_reference_tokens,
        }

