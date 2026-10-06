# IA Doblaje

Subtitulado, traducción multidioma y subtítulos en vivo.

## Motor

```text
python -m app.presentation.cli analyze video.mp4 --profile balanced --device cpu --output data/output
python -m app.presentation.api
```

La API escucha en `http://127.0.0.1:8000`.

## Web

El frontend está en `../video_subtitles`.

```text
cd ../video_subtitles
npm run dev
```

Next.js reescribe `/api` hacia `http://127.0.0.1:8000`.

## Requisitos

| Componente | Versión verificada | Notas |
|---|---|---|
| Python | 3.14.7 | `>=3.11,<3.15` |
| FFmpeg / FFprobe | 9.0.1 | Se localiza aunque no esté en el `PATH` (WinGet Links + registro) |
| Node.js | 24.x | Frontend Next.js 16 |

```text
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,asr,diarization,vad]"
```

## Modelos locales

Los modelos **no se descargan en tiempo de ejecución**. Deben existir en `models/`:

| Paso | Archivo | Cómo obtenerlo |
|---|---|---|
| T04 VAD | `silero_vad_v5.1.jit` | `python -c "import torch; torch.hub.load('snakers4/silero-vad','silero_vad')"` y copiar `silero_vad.jit` desde la caché de torch |
| T05 LID | `whisper_encoder_small_multilingual_lid_v1.jit` | `python scripts/export_whisper_lid_jit.py --checkpoint models/whisper_small_pytorch/small.pt` |
| T06 ASR | `whisper_small_ct2_int8_fp16_local_v1/model.bin` | Modelo CTranslate2 local |
| T08 Diarización | `pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin` | `huggingface_hub.hf_hub_download` sobre `pyannote/wespeaker-voxceleb-resnet34-LM` |
| T15 Traducción | `m2m100_418M/pytorch_model.bin` | Pesos M2M100 418M |

Comprobación del estado de preparación:

```text
GET /api/system/preflight
```

## Pruebas

```text
# Motor + API (desde la raíz del repo)
.venv\Scripts\python -m pytest tests -q

# Frontend
cd ../video_subtitles
npm run verify     # typecheck + lint + tests + build
```

## Documentación

- `docs/PIPELINE.md` — pasos T01–T15.
- `docs/ARCHITECTURE.md` — capas domain/application/infrastructure/presentation.
- `docs/RENDERING.md` — patrones de render del frontend.
- `docs/AUDITORIA_ESTADO.md` — auditoría de estado y plan.
