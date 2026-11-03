from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401 — usado en forward de _WhisperLidInner
import torch.nn as nn  # noqa: F401 — usado por _WhisperLidInner/_WhisperLidOuter (nivel módulo)

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_SAMPLES = 480000
SAMPLE_RATE = 16000
N_AUDIO_CTX = 1500
D_MODEL = 768
NUM_CLASSES = 99

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
        audio = audio.unsqueeze(0)
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
        self.register_buffer("sot_token", torch.tensor([[sot_token]], dtype=torch.long))
        self.register_buffer("lang_token_ids", lang_token_ids.detach().clone().to(torch.long))
        self.register_buffer("mel_filters", mel_filters.detach().clone())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = _pad_or_trim(audio, N_SAMPLES)
        mel = _log_mel(audio, self.mel_filters)
        feats = self.encoder(mel)
        logits = self.decoder(self.sot_token, feats)[:, 0]
        lang_logits = logits[0].index_select(0, self.lang_token_ids)
        lang_logits = lang_logits.unsqueeze(0)
        sentinel = torch.full_like(lang_logits[:, :1], float(SENTINEL_LOGIT))
        return torch.cat([lang_logits, sentinel], dim=1)


class _WhisperLidOuter(nn.Module):
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

    from whisper import model as _wm

    _wm.MultiHeadAttention.use_sdpa = False

    t0 = time.perf_counter()
    model = whisper.load_model("small", download_root=str(ckpt_path.parent))
    print(f"      whisper versión: {whisper.__version__}")
    print(f"      dims: {model.dims}")
    print(f"[2/7] Extrayendo language token IDs (orden exacto del proyecto)")

    encoding = get_encoding(name="multilingual", num_languages=99)
    lang_token_ids: list[int] = []
    for code in PROJECT_LANG_CODES:
        tok = f"<|{code}|>"
        tid = encoding.encode_single_token(tok)
        lang_token_ids.append(tid)
    assert len(lang_token_ids) == NUM_CLASSES - 1, "debe haber 98 language tokens"
    lang_ids_tensor = torch.tensor(lang_token_ids, dtype=torch.long)

    sot_token: int = int(encoding.encode_single_token("<|startoftranscript|>"))
    print(f"      sot_token={sot_token}, lang_tokens={len(lang_token_ids)}")

    print("[3/7] Obteniendo filtros mel oficiales")
    from whisper.audio import mel_filters as _mel_filters

    mel_filters = _mel_filters(torch.device("cpu"), N_MELS).detach().clone()

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

    example_audio = torch.randn(1, SAMPLE_RATE * 3, dtype=torch.float32)
    with torch.inference_mode():
        traced_inner = torch.jit.trace(inner, example_audio, check_trace=False)

    outer = _WhisperLidOuter(traced_inner)
    outer.eval()
    with torch.inference_mode():
        scripted = torch.jit.script(outer)

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
