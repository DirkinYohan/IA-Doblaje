"""Exportar el modelo LID T05 a TorchScript JIT — RECONSTRUCCIÓN (RUTA B).

Este script NO entrena y NO descarga datasets. Reconstruye un modelo
funcionalmente compatible con el contrato de
`WhisperEncoderLanguageDetectionAdapter` usando únicamente los pesos
OFICIALES MIT de `openai/whisper-small`.

----------------------------------------------------------------------
CONTRATO QUE DEBE CUMPLIR (adapter T05)
----------------------------------------------------------------------
    forward(audio, sr) -> logits [B, 99]
      audio : torch.Tensor float32, shape [1, N], rango [-1, 1]
      sr    : int = 16000
      logits: torch.Tensor shape [1, 99] (99 clases de idioma, orden
              EXACTO del proyecto WHISPER_MULTILINGUAL_LANGUAGES)

Propiedades exigidas:
    - TorchScript JIT (torch.jit.load)
    - PyTorch 2.4.1+cpu
    - 100% offline tras la exportación (sin HF, sin red)
    - tamaño > 100 MB (guard del loader)

----------------------------------------------------------------------
ORIGEN EXACTO DE PESOS
----------------------------------------------------------------------
    - Checkpoint: openai/whisper-small  ("small.pt", 461 MB)
      URL oficial: https://openaipublic.azureedge.net/main/whisper/models/
                   9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
    - Descargado con openai-whisper 20250625 (whisper.load_model('small'))
    - Código: openai/whisper (MIT)
    - Pesos: MIT

----------------------------------------------------------------------
ARQUITECTURA (reconstrucción)
----------------------------------------------------------------------
    audio [1, N]  (float32, [-1,1])
        -> pad/trim a 480000 muestras (30 s)   [1, 480000]
        -> log-mel spectrogram 80 mels          [1, 80, 3000]
              (hann 400, hop 160, clamp 1e-10, log10,
               recorte -8 dB, escalado (x+4)/4)  == whisper.audio
        -> Whisper Small ENCODER                 [1, 1500, 768]
              (d_model=768, 12 capas, 12 cabezas, d_model=768)
        -> mean pooling temporal                 [1, 768]
        -> proyección lineal SIN bias:
              logits = pooled @ W_lang.T         [1, 99]
              W_lang[0:98] = decoder.token_embedding.weight[lang_token_ids]
                              (embeddings oficiales de los 98 language tokens)
              W_lang[98]   = logit muy negativo (clase sentinel 'und')

NOTA DE HONESTIDAD:
    Este artefacto es una RECONSTRUCCIÓN funcionalmente compatible. NO es
    idéntico al "whisper_encoder_small_multilingual_lid_v1.jit" original
    (irrecuperable). Usa el encoder oficial y los embeddings oficiales de
    los language tokens del checkpoint MIT; la única decisión de diseño
    propia es el mean-pooling + proyección lineal sobre dichos embeddings
    (fiel a la descripción "Whisper encoder -> probabilidades multiclase"
    de docs/PIPELINE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401 — usado en forward de _WhisperLidInner
import torch.nn as nn  # noqa: F401 — usado por _WhisperLidInner/_WhisperLidOuter (nivel módulo)

# ---------------------------------------------------------------------------
# CONSTANTES (fijas, igual que whisper.audio y docs/PIPELINE.md)
# ---------------------------------------------------------------------------
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_SAMPLES = 480000  # 30 s @ 16 kHz
SAMPLE_RATE = 16000
N_AUDIO_CTX = 1500  # frames del encoder
D_MODEL = 768
NUM_CLASSES = 99

# Mapa del proyecto (orden EXACTO): índice -> ISO code. Índice 98 = sentinel 'und'.
PROJECT_LANG_CODES: tuple[str, ...] = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "la", "mi", "ml", "cy", "sk", "te",
    "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
    "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl",
    "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka",
    "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht",
    "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
    "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
)
SENTINEL_LOGIT = -1e9


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pad_or_trim(audio: "torch.Tensor", length: int = N_SAMPLES) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # [1, N]
    n = audio.shape[1]
    if n > length:
        audio = audio[:, :length]
    elif n < length:
        audio = F.pad(audio, (0, length - n))
    return audio


def _log_mel(audio: "torch.Tensor", filters: "torch.Tensor") -> "torch.Tensor":
    import torch

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = filters.to(audio.device) @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class _WhisperLidInner(nn.Module):
    """Módulo interno (solo entrada tensorial) — se exporta por trace.

    Mecanismo LID REAL de Whisper (idéntico a `Whisper.detect_language`):
        audio -> mel -> ENCODER (audio features)
        logits = DECODER(token startoftranscript, audio_features)[:, 0]
        language_logits = logits[index language tokens] -> [98]
        + sentinel 'und' -> [99]
    """

    def __init__(
        self,
        encoder,
        decoder,
        sot_token: int,
        lang_token_ids: "torch.Tensor",
        mel_filters: "torch.Tensor",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # buffers persistentes -> quedan dentro del JIT
        self.register_buffer("sot_token", torch.tensor([[sot_token]], dtype=torch.long))  # [1, 1]
        self.register_buffer("lang_token_ids", lang_token_ids.detach().clone().to(torch.long))  # [98]
        self.register_buffer("mel_filters", mel_filters.detach().clone())  # [80, 201]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = _pad_or_trim(audio, N_SAMPLES)  # [1, 480000]
        mel = _log_mel(audio, self.mel_filters)  # [1, 80, 3000]
        feats = self.encoder(mel)  # [1, 1500, 768]
        # Un paso del decoder sobre el token startoftranscript (cross-attention)
        logits = self.decoder(self.sot_token, feats)[:, 0]  # [1, n_vocab]
        lang_logits = logits[0].index_select(0, self.lang_token_ids)  # [98]
        lang_logits = lang_logits.unsqueeze(0)  # [1, 98]
        # Clase 98 (sentinel 'und'): logit constante muy negativo (nunca argmax).
        sentinel = torch.full_like(lang_logits[:, :1], float(SENTINEL_LOGIT))
        return torch.cat([lang_logits, sentinel], dim=1)  # [1, 99]


class _WhisperLidOuter(nn.Module):
    """Módulo externo scripted: firma forward(audio, sr) exigida por el adapter.

    `sr` se conserva por compatibilidad de firma; el modelo opera a 16 kHz fijo.
    """

    def __init__(self, inner: "torch.nn.Module") -> None:
        super().__init__()
        self.inner = inner

    def forward(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        return self.inner(audio)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description="Export Whisper-LID TorchScript (Ruta B)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/whisper_small_pytorch/small.pt"),
        help="Ruta al checkpoint openai/whisper-small (small.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/whisper_encoder_small_multilingual_lid_v1.jit"),
        help="Ruta de salida del TorchScript",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path("models/whisper_encoder_small_multilingual_lid_v1.meta.json"),
        help="Ruta del JSON de trazabilidad",
    )
    args = parser.parse_args()

    ckpt_path: Path = args.checkpoint.resolve()
    out_path: Path = args.output.resolve()
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint no encontrado: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[1/7] Cargando checkpoint: {ckpt_path}")
    print(f"      SHA256 checkpoint: {_sha256_file(ckpt_path)}")

    import whisper
    from whisper.tokenizer import get_encoding

    # Desactivar SDPA durante la exportación: `scaled_dot_product_attention`
    # con `is_causal=mask is not None` se vuelve Tensor al hacer trace y falla.
    # Forzar la ruta manual de atención (matemáticamente equivalente).
    from whisper import model as _wm

    _wm.MultiHeadAttention.use_sdpa = False

    t0 = time.perf_counter()
    model = whisper.load_model("small", download_root=str(ckpt_path.parent))
    print(f"      whisper versión: {whisper.__version__}")
    print(f"      dims: {model.dims}")
    print(f"[2/7] Extrayendo language token IDs (orden exacto del proyecto)")

    encoding = get_encoding(name="multilingual", num_languages=99)
    # token ID por CÓDIGO (fiel al orden del proyecto; evita el desfase por 'lt')
    lang_token_ids: list[int] = []
    for code in PROJECT_LANG_CODES:
        tok = f"<|{code}|>"
        tid = encoding.encode_single_token(tok)
        lang_token_ids.append(tid)
    assert len(lang_token_ids) == NUM_CLASSES - 1, "debe haber 98 language tokens"
    lang_ids_tensor = torch.tensor(lang_token_ids, dtype=torch.long)  # [98]

    # Token startoftranscript (inicio del prompt LID, igual que detect_language)
    sot_token: int = int(encoding.encode_single_token("<|startoftranscript|>"))
    print(f"      sot_token={sot_token}, lang_tokens={len(lang_token_ids)}")

    print("[3/7] Obteniendo filtros mel oficiales")
    from whisper.audio import mel_filters as _mel_filters

    mel_filters = _mel_filters(torch.device("cpu"), N_MELS).detach().clone()  # [80, 201]

    print("[4/7] Construyendo wrapper y exportando TorchScript (trace + script)")
    import torch.nn as nn

    inner = _WhisperLidInner(
        model.encoder,
        model.decoder,
        sot_token,
        lang_ids_tensor,
        mel_filters,
    )
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)

    example_audio = torch.randn(1, SAMPLE_RATE * 3, dtype=torch.float32)  # 3 s
    with torch.inference_mode():
        traced_inner = torch.jit.trace(inner, example_audio, check_trace=False)

    # Envolver en módulo scripted con firma forward(audio, sr)
    outer = _WhisperLidOuter(traced_inner)
    outer.eval()
    with torch.inference_mode():
        scripted = torch.jit.script(outer)

    # optimización de inferencia
    scripted = torch.jit.freeze(scripted)
    scripted.save(str(out_path))
    print(f"      Exportado: {out_path}")

    print("[5/7] Validando output del TorchScript")
    loaded = torch.jit.load(str(out_path), map_location="cpu")
    loaded.eval()
    with torch.inference_mode():
        out = loaded(torch.randn(1, SAMPLE_RATE * 2, dtype=torch.float32), SAMPLE_RATE)
    print(f"      output type: {type(out).__name__}")
    print(f"      output shape: {tuple(out.shape)}")
    assert tuple(out.shape) == (1, NUM_CLASSES), "shape != [1, 99]"
    assert out.dtype == torch.float32
    out_np = out.detach().cpu().numpy()
    assert not bool((out_np != out_np).any()), "contiene NaN"
    assert not bool((abs(out_np) == float("inf")).any()), "contiene Inf"

    # softmax valida y top-k
    logits = out_np[0]
    logits_stable = logits - float(logits.max())
    exp = __import__("numpy").exp(logits_stable)
    probs = exp / float(exp.sum())
    assert abs(float(probs.sum()) - 1.0) < 1e-4, "softmax no suma 1"
    top1 = int(logits.argmax())
    print(f"      top-1 idx: {top1} -> {PROJECT_LANG_CODES[top1]}")
    assert 0 <= top1 < 99
    print("      VALIDACIÓN OUTPUT: PASS")

    print("[6/7] Escribiendo metadata de trazabilidad")
    meta = {
        "artifact": "whisper_encoder_small_multilingual_lid_v1.jit",
        "kind": "RECONSTRUCCION_RUTA_B_NO_ENTRENAMIENTO",
        "honestidad": (
            "Reconstruccion funcionalmente compatible. NO es identico al v1 original "
            "irrecuperable. Encoder + DECODER oficiales MIT (mecanismo LID real de "
            "Whisper: token startoftranscript + cross-attention)."
        ),
        "modelo_base": "openai/whisper-small",
        "base_license": "MIT",
        "whisper_version": whisper.__version__,
        "torch_version": torch.__version__,
        "checkpoint_sha256": _sha256_file(ckpt_path),
        "arquitectura": {
            "encoder": "Whisper Small encoder (d_model=768, 12 capas, 12 cabezas)",
            "decoder": "Whisper Small decoder (1 paso cross-attention con token startoftranscript)",
            "frontend": "log-mel 80 mels (hann 400, hop 160) == whisper.audio",
            "head": "logits del decoder indexados en 98 language tokens + sentinel 'und'",
            "num_clases": NUM_CLASSES,
            "sample_rate": SAMPLE_RATE,
            "input": "float32 [1, N], rango [-1,1]",
            "output": "logits [1, 99]",
        },
        "mapa_99_idiomas": list(PROJECT_LANG_CODES),
        "sentinel": {"indice": 98, "codigo": "und", "logit": SENTINEL_LOGIT},
        "procedimiento": (
            "1) whisper.load_model('small') descarga small.pt oficial; "
            "2) encoder(mel) -> audio features; decoder(token sot, features) -> logits; "
            "3) indexar logits en language tokens (98) + sentinel 'und'; "
            "4) torch.jit.trace con ejemplo de 3 s; torch.jit.freeze; save."
        ),
        "output_sha256": None,
        "output_size_bytes": None,
        "fecha_exportacion_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    out_stat = out_path.stat()
    meta["output_sha256"] = _sha256_file(out_path)
    meta["output_size_bytes"] = out_stat.st_size
    args.meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      Metadata: {args.meta_out}")

    print("[7/7] LISTO")
    print(f"      SHA256 artifact: {meta['output_sha256']}")
    print(f"      Size: {meta['output_size_bytes']} bytes "
          f"({meta['output_size_bytes']/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
