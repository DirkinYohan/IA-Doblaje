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
            from app.core.cpu import thread_count

            self._engine = ctranslate2.Translator(
                str(self.root / CT2_FOLDERNAME), device="cpu", compute_type="int8",
                intra_threads=thread_count(),
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
            tokens = result.hypotheses[0][1:]
            ids = tokenizer.convert_tokens_to_ids(tokens)
            out.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
        return out


_SHARED: M2M100CT2Translator | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> M2M100CT2Translator:
    """Una sola instancia por proceso: el modelo ocupa memoria y no debe cargarse dos veces."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = M2M100CT2Translator()
        return _SHARED


__all__ = ["CT2_FOLDERNAME", "M2M100CT2Translator", "is_available", "models_dir", "shared"]
