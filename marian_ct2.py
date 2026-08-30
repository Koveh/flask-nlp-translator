from __future__ import annotations

import logging
import os
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

import ctranslate2
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., object])

DEFAULT_BEAM_SIZE = 4
DEFAULT_MAX_DECODING_LENGTH = 512


def logged(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        LOGGER.info("%s", fn.__name__)
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class MarianCt2Translator:
    """CPU CTranslate2 INT8 runtime for Helsinki OPUS-MT (Marian) hubs."""

    engine = "ctranslate2-int8"

    def __call__(self, text: str, **kwargs) -> list[dict[str, str]]:
        return [{"translation_text": self.translate(text)}]

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_id: str,
        beam_size: int = DEFAULT_BEAM_SIZE,
        compute_type: str = "int8",
        intra_threads: int | None = None,
        max_decoding_length: int = DEFAULT_MAX_DECODING_LENGTH,
    ) -> None:
        threads = intra_threads if intra_threads is not None else min(8, os.cpu_count() or 4)
        self._beam_size = beam_size
        self._max_decoding_length = max_decoding_length
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._translator = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            compute_type=compute_type,
            intra_threads=threads,
            inter_threads=1,
        )

    def _decode(self, tokens: list[str]) -> str:
        ids = self._tokenizer.convert_tokens_to_ids(tokens)
        return self._tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _source_tokens(self, text: str) -> list[str]:
        # Marian CT2 config has add_source_eos=false; keep tokenizer </s> or the model repeats the sentence.
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        return self._tokenizer.convert_ids_to_tokens(ids)

    def translate(self, text: str) -> str:
        results = self._translator.translate_batch(
            [self._source_tokens(text)],
            beam_size=self._beam_size,
            max_decoding_length=self._max_decoding_length,
        )
        return self._decode(results[0].hypotheses[0])
