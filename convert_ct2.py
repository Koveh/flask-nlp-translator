#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

from ctranslate2.converters.transformers import TransformersConverter

from marian_ct2 import logged

LOGGER = logging.getLogger(__name__)

HUB_GMW_DEU_ENG_NLD = "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_nld"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "ct2-gmw-int8"


class MarianTransformersConverter(TransformersConverter):
    """CTranslate2 4.8.1 passes dtype= into MarianMTModel; transformers 4.53 rejects it."""

    def load_model(self, model_class, model_name_or_path, **kwargs):
        kwargs.pop("dtype", None)
        kwargs.pop("torch_dtype", None)
        return model_class.from_pretrained(model_name_or_path, **kwargs)


@logged
def convert_hub(hub_id: str, output_dir: Path, quantization: str = "int8") -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    converter = MarianTransformersConverter(hub_id)
    converter.convert(str(output_dir), quantization=quantization, force=True)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = convert_hub(HUB_GMW_DEU_ENG_NLD, DEFAULT_OUTPUT)
    LOGGER.info("converted %s -> %s", HUB_GMW_DEU_ENG_NLD, path)


if __name__ == "__main__":
    main()
