from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WHISPER_REPO = "Systran/faster-whisper-small"
WHISPER_FOLDER = "whisper_small_ct2_int8_fp16_local_v1"
M2M100_REPO = "facebook/m2m100_418M"
M2M100_FOLDER = "m2m100_418M"
M2M100_CT2_FOLDER = "m2m100_418M_ct2_int8"
M2M100_FILES = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]


def fetch_whisper(models: Path) -> None:
    from huggingface_hub import snapshot_download

    target = models / WHISPER_FOLDER
    if (target / "model.bin").is_file():
        print(f"Whisper ya está en {target}")
        return
    snapshot_download(WHISPER_REPO, local_dir=str(target))
    print(f"Whisper descargado en {target}")


def fetch_m2m100(models: Path, keep_weights: bool) -> None:
    from huggingface_hub import snapshot_download

    source = models / M2M100_FOLDER
    target = models / M2M100_CT2_FOLDER
    if (target / "model.bin").is_file() and (source / "sentencepiece.bpe.model").is_file():
        print(f"M2M100 ya está en {target}")
        return
    snapshot_download(M2M100_REPO, local_dir=str(source), allow_patterns=M2M100_FILES)
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(str(source)).convert(str(target), quantization="int8", force=True)
    print(f"M2M100 convertido en {target}")
    if not keep_weights:
        (source / "pytorch_model.bin").unlink(missing_ok=True)
        shutil.rmtree(source / ".cache", ignore_errors=True)
    shutil.rmtree(models / WHISPER_FOLDER / ".cache", ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga los modelos que usa la API.")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--keep-weights", action="store_true", help="Conserva los pesos PyTorch de M2M100.")
    args = parser.parse_args()
    models = args.models_dir.resolve()
    models.mkdir(parents=True, exist_ok=True)
    fetch_whisper(models)
    fetch_m2m100(models, args.keep_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
