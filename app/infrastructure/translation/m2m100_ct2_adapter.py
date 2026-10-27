"""M2M100 418M sobre CTranslate2 (int8, CPU). Offline.

Misma interfaz de lote que ``M2M100TranslatorAdapter.translate_batch`` pero con
el motor de inferencia que ya usa Whisper. Medido en este equipo (CPU, 12
hilos) con tres frases en→es: carga 1,5 s frente a 12,4 s y traducción 0,65 s
frente a 2,6 s, con el mismo resultado. Es lo que permite traducir por delante
de la reproducción en el modo en vivo.

El modelo convertido se genera con ``scripts/convert_m2m100_ct2.py`` a partir
del modelo local ``models/m2m100_418M`` (de él también sale el tokenizador).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

CT2_FOLDERNAME = "m2m100_418M_ct2_int8"
TOKENIZER_FOLDERNAME = "m2m100_418M"


def models_dir() -> Path:
    from app.core.config import get_settings

    return Path(get_settings().paths.models_cache_dir).expanduser().resolve()


def is_available(root: Path | None = None) -> bool:
    base = root or models_dir()
    return (base / CT2_FOLDERNAME / "model.bin").is_file() and (
        base / TOKENIZER_FOLDERNAME / "sentencepiece.bpe.model"
    ).is_file()


class M2M100CT2Translator:
    """Traductor por lotes. No es reentrante: el llamador serializa."""

    def __init__(self, root: Path | None = None, *, max_length: int = 128) -> None:
        self.root = root or models_dir()
        self.max_length = int(max_length)
        self._lock = threading.Lock()
        self._engine: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        with self._lock:
            if self._engine is not None:
                return
            import ctranslate2
            from transformers import M2M100Tokenizer

            self._tokenizer = M2M100Tokenizer.from_pretrained(
                str(self.root / TOKENIZER_FOLDERNAME), local_files_only=True
            )
            self._engine = ctranslate2.Translator(
                str(self.root / CT2_FOLDERNAME), device="cpu", compute_type="int8"
            )

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        if not texts:
            return []
        self.load()
        tokenizer = self._tokenizer
        tokenizer.src_lang = source
        batch = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts]
        prefix = [tokenizer.lang_code_to_token[target]]
        results = self._engine.translate_batch(
            batch,
            target_prefix=[prefix] * len(batch),
            beam_size=1,
            max_decoding_length=self.max_length,
        )
        out: list[str] = []
        for result in results:
            tokens = result.hypotheses[0][1:]  # sin la etiqueta de idioma
            ids = tokenizer.convert_tokens_to_ids(tokens)
            out.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
        return out


__all__ = ["CT2_FOLDERNAME", "M2M100CT2Translator", "is_available", "models_dir"]
