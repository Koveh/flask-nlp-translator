from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import sentencepiece as spm

DEFAULT_BEAM_SIZE = 4
DEFAULT_MAX_DECODING_LENGTH = 512


def _translator_class() -> type:
    # ctranslate2/__init__.py imports converters, which pulls torch. Stub those
    # submodules so the CPU runtime stays SentencePiece + the C extension.
    for name in ("ctranslate2.converters", "ctranslate2.models", "ctranslate2.specs"):
        sys.modules.setdefault(name, types.ModuleType(name))
    from ctranslate2 import Translator

    return Translator


Translator = _translator_class()


class MarianCt2Translator:
    """CPU CTranslate2 INT8 runtime for Helsinki OPUS-MT (Marian) hubs."""

    engine = "ctranslate2-int8"

    def __call__(self, text: str, **kwargs) -> list[dict[str, str]]:
        return [{"translation_text": self.translate(text)}]

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_id: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        compute_type: str = "int8",
        intra_threads: int | None = None,
        max_decoding_length: int = DEFAULT_MAX_DECODING_LENGTH,
    ) -> None:
        del tokenizer_id
        model_path = Path(model_dir)
        threads = intra_threads if intra_threads is not None else min(8, os.cpu_count() or 4)
        self._beam_size = beam_size
        self._max_decoding_length = max_decoding_length
        self._source_spm = spm.SentencePieceProcessor(model_file=str(model_path / "source.spm"))
        self._target_spm = spm.SentencePieceProcessor(model_file=str(model_path / "target.spm"))
        self._translator = Translator(
            str(model_path),
            device="cpu",
            compute_type=compute_type,
            intra_threads=threads,
            inter_threads=1,
        )

    def _decode(self, tokens: list[str]) -> str:
        return self._target_spm.decode(tokens).strip()

    def _source_tokens(self, text: str) -> list[str]:
        tag, body = self._split_target_tag(text)
        pieces = self._source_spm.encode(body, out_type=str)
        # Marian CT2 config has add_source_eos=false; keep </s> or the model repeats the sentence.
        return ([tag] if tag else []) + pieces + ["</s>"]

    def _split_target_tag(self, text: str) -> tuple[str | None, str]:
        if text.startswith(">>") and "<<" in text[:16]:
            end = text.index("<<") + 2
            return text[:end], text[end:].lstrip()
        return None, text

    def translate(self, text: str) -> str:
        results = self._translator.translate_batch(
            [self._source_tokens(text)],
            beam_size=self._beam_size,
            max_decoding_length=self._max_decoding_length,
        )
        return self._decode(results[0].hypotheses[0])
