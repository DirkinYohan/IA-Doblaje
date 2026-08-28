# 🎙️ IA-DOBLAJE ENGINE

> **Motor Profesional de IA para Doblaje Automático** — Fase 1: Análisis Inteligente de Audio

---

## 🚧 ESTADO DEL PROYECTO: FASE 1 — PRE-ALPHA

El proyecto se encuentra actualmente en **Fase 1 (Análisis de Audio)**. Las fases futuras (Traducción, TTS, Clonación, Sincronización Labial, Mezcla, Render Final) están **planeadas pero NO implementadas**.

### Fases del roadmap
```
Fase 1  [EN CURSO]  Audio Intelligence Engine (ASR + Diar + Timestamps + Quality)
Fase 2  [      ]    Translation Engine
Fase 3  [      ]    Dialogue Adaptation
Fase 4  [      ]    Voice Generation (TTS)
Fase 5  [      ]    Voice Character Management (Speaker Identity)
Fase 6  [      ]    Temporal Synchronization
Fase 7  [      ]    Lip Synchronization
Fase 8  [      ]    Audio Mixing
Fase 9  [      ]    Final Video Rendering
```

---

## 📋 QUÉ HACE LA FASE 1 (IMPLEMENTADO O EN IMPLEMENTACIÓN)

Dado un archivo multimedia de entrada (MP4, MKV, MOV, WAV, MP3, M4A):

```
python -m app analyze data/input/pelicula.mp4 --profile balanced --output data/output/
```

El motor produce una **representación estructurada validada** de todo el diálogo:

| Salida | Archivo JSON | Contenido |
|---|---|---|
| Análisis completo | `analysis.json` | Agregado root con media + processing + language + speakers + silences + segments_aligned + quality + metrics + validation |
| Transcripción | `transcript.json` | Transcripción centrada en texto/palabras/timestamps |
| Hablantes | `speakers.json` | Diarización raw (SpeakerTurns SPEAKER_XX SIN identidad personaje) |
| Segmentos alineados | `segments.json` | DialogueSegment alineados ASR↔Diarization |
| Metadatos | `metadata.json` | Media metadata + processing info + métricas agregadas |

### Capacidades concretas
- ✅ Detección automática de formato y validación de archivos
- ✅ Extracción de audio normalizada (16 kHz mono PCM via FFmpeg)
- ✅ Voice Activity Detection (Silero VAD, PyTorch)
- ✅ Detección de idioma 99+ lenguajes
- ✅ Speech-to-Text con timestamps a nivel segmento y palabra
- ✅ Speaker Diarization (SOLO SPEAKER_XX, SIN identidad de personaje)
- ✅ Alineación por solape pesado ASR ↔ Diarization
- ✅ **Quality Analysis** con 11 reglas + score 0.0–1.0
- ✅ **Metrics Service**: RTF, memoria CPU/GPU, utilidades, tiempos por step
- ✅ Perfiles `quality` / `balanced` / `performance` (modelo + adapter + compute_type)
- ✅ No hay downgrade silencioso de perfil — confirmación explícita
- ✅ Safety: validación paths, size limits, temp dir UUID, subprocess seguro

---

## 🏗️ ARQUITECTURA

Clean Architecture adaptada, cumplimiento SOLID, Dependency Inversion via ABC Interfaces.

```
┌────────────────────────────────────────────────────┐
│ DOMAIN (Entities, ValueObjects, ABC Interfaces)    │ Sin dependencias IA
├────────────────────────────────────────────────────┤
│ APPLICATION (UseCases, Services, Quality, Metrics) │ Depende solo de Domain
├────────────────────────────────────────────────────┤
│ INFRASTRUCTURE (ASR adapters, VAD, FFmpeg, etc.)   │ Implementa Domain ABCs
├────────────────────────────────────────────────────┤
│ PRESENTATION (CLI Typer + FastAPI skeleton)        │ Entrypoints
└────────────────────────────────────────────────────┘
```

Documentación detallada en:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PIPELINE.md](docs/PIPELINE.md) — 14 pasos del pipeline
- [docs/QUALITY_RULES.md](docs/QUALITY_RULES.md) — 11 reglas QR01-QR11
- [docs/METRICS.md](docs/METRICS.md) — diccionario de métricas
- [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md) — ⚠️ Cumplimiento legal
- [docs/LICENSES_ANALYSIS.md](docs/LICENSES_ANALYSIS.md) — ⚠️ Análisis riesgo + componentes REQUIERE REVISIÓN LEGAL

---

## 📋 PREREQUISITOS

| Requisito | Versión mínima | Razón |
|---|---|---|
| Python | 3.11.x | Type hints modernas, `tomllib` stdlib |
| PyTorch | 2.3.x+ | Framework IA central |
| FFmpeg | 6.0 LGPL build | Extracción audio |
| (Opcional) CUDA Toolkit | 11.8+ | Aceleración GPU NVIDIA |
| (Opcional) HuggingFace Account + Token | — | Descarga pesos pyannote gated |

### Requisitos GPU recomendados por perfil
| Perfil | VRAM mínima | Nota |
|---|---|---|
| **PERFORMANCE** | 3 GB | faster-whisper large-v3 int8 |
| **BALANCED** | 5 GB | whisper-pytorch medium FP16 |
| **QUALITY** | 8 GB | whisper-pytorch large-v3 FP16 + beam_size=5 |

> ⚠️ **Política anti-downgrade silencioso (PUNTO 2):** Si seleccionas `QUALITY` y tu GPU no alcanza la VRAM requerida, el CLI te preguntará explícitamente qué hacer. **Nunca** se degradará el perfil sin tu consentimiento y el downgrade quedará registrado en `metrics.profile_downgrade_applied`.

---

## 🚀 INSTALACIÓN

```powershell
# 1. Clonar repo
cd "c:\Ruta\IA Doblaje"

# 2. Crear entorno virtual (opcional, recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Instalar modo desarrollo con herramientas de dev + quality
pip install -e ".[dev,quality]"

# 4. Copiar variables de entorno
Copy-Item .env.example .env
# Editar .env: ajustar PROFILE, DEVICE, HF_TOKEN (si usas pyannote)

# 5. Verificar instalación
python -m app --version
python -m app --help
```

---

## 🔧 USO RÁPIDO — CLI

```bash
# Ver comandos
python -m app --help

# Ver versiones + diagnostico sistema
python -m app diagnose

# Analizar video (fase 1 completo)
python -m app analyze data/input/prueba_30s.mp4 ^
    --profile balanced ^
    --device auto ^
    --output data/output/ ^
    --job-id mi_analisis_001

# Analizar con perfil de maxima calidad
python -m app analyze data/input/pelicula.mp4 --profile quality

# Validar un JSON de salida existente
python -m app validate data/output/analysis.json
```

---

## ✅ TESTING

```bash
# Ejecutar todos los tests (unit + integracion, salvo @slow)
pytest

# Solo tests unitarios
pytest tests/domain tests/application -v

# Con coverage
pytest --cov=app --cov-report=html:htmlcov

# Ejecutar tambien tests lentos de integracion
pytest --run-slow -m integration
```

---

## 🔍 CALIDAD DE CÓDIGO

```bash
# Linting + formato (Ruff)
ruff check app tests scripts
ruff format --check app tests scripts

# Tipado estatico (mypy)
mypy app

# Seguridad (bandit)
bandit -c pyproject.toml -r app
```

---

## ⚠️ CUMPLIMIENTO LICENCIAS (OBLIGATORIO ANTES DE USO COMERCIAL)

**NO USE ESTE SOFTWARE EN PRODUCCIÓN SIN ANTES LEER:**

1. 👉 [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md)
    - Lista completa dependencia × versión × licencia código × licencia pesos × TOS × restricciones redistribución × atribución
    - Columnas explícitas: `Uso comercial`, `REQUIERE REVISIÓN LEGAL`

2. 👉 [docs/LICENSES_ANALYSIS.md](docs/LICENSES_ANALYSIS.md)
    - Análisis de riesgo componente a componente
    - Instrucciones build FFmpeg **LGPL-ONLY** (sin x264/x265/fdk-aac GPL)
    - Procedimiento aceptación HF gated models (pyannote)
    - CUDA/cuDNN: dynamic-link + NO redistribución
    - Script de verificación automática: `scripts/verify_third_party.py`

Cualquier duda legal contacte al responsable del proyecto. **NO asuma que una dependencia es comercialmente utilizable.**

---

## 📦 DOCKER (Futuro, T13)

La imagen oficial PyTorch + FFmpeg LGPL build está planificada para T13 de la Fase 1.

```bash
# No implementado aun en T01
# docker compose build
# docker compose run --rm analyze /data/input/test.mp4 --profile balanced
```

---

## 🧩 ESTRUCTURA DEL PROYECTO (T01 Completado)

```
ia-doblaje-engine/
├── app/
│   ├── core/                          # Cross-cutting (config, logging, device, exceptions)
│   ├── domain/                        # Entities + ValueObjects + ABC Interfaces
│   │   ├── entities/
│   │   ├── value_objects/
│   │   └── interfaces/
│   ├── application/                   # UseCases + Services (Alignment, Quality, Metrics, Validation, Serialization)
│   │   ├── use_cases/
│   │   ├── services/quality/          # ← Modulo Quality Analysis PUNTO 4
│   │   └── pipeline/
│   ├── infrastructure/                # Implementaciones de Interfaces
│   │   ├── audio/
│   │   ├── asr/
│   │   ├── diarization/
│   │   ├── language/
│   │   └── storage/
│   └── presentation/cli/              # Typer CLI entrypoints
├── tests/                             # tests por capa + integracion progresiva 30s/1min/5min/30min
├── scripts/                           # utils dev/download/verify
├── data/{input,output,temporary}/     # working dirs (gitkeep, no data)
├── configs/                           # quality.yaml, balanced.yaml, performance.yaml (plan T02)
├── docs/                              # ARCHITECTURE, PIPELINE, QUALITY, METRICS, LICENCIAS
├── models/                            # HF cache local (gitignore, no commit)
├── pyproject.toml                     # PEP 621, deps, pytest, ruff, mypy, bandit
├── .env.example
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

---

## 🎯 CRITERIO DE FINALIZACIÓN FASE 1

Se considerará la Fase 1 completada cuando un vídeo de prueba real (≥ 1 minuto) produzca de forma **reproducible** y **validada**:

```
✓ Media loaded + validated
✓ Audio extracted (16kHz mono PCM)
✓ Language detected (+ confidence)
✓ Speech recognized
✓ Segment + Word timestamps
✓ Speakers detected (SPEAKER_XX SOLO diarización)
✓ Dialogue aligned (speaker ↔ texto ↔ TS)
✓ Quality Analysis report con score 0-1
✓ Metrics: RTF < 1.0 GPU, memoria < límites perfil
✓ 5 JSON outputs validados contra schema Pydantic
✓ Todos los tests unit + integration (≥30s clip) pasan
✓ docs/THIRD_PARTY_LICENSES.md verificación humana firmada
```

---

## 🚧 Siguiente paso

> [T02 en espera] — Core Layer: `config.py` Pydantic Settings con Profiles, DeviceDetector, Exceptions Hierarchy, SafePathResolver.

---

## 📄 Licencia del proyecto

Ver el archivo [LICENSE](LICENSE) para los términos del código propietario del proyecto.  
Ver [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md) para el cumplimiento de licencias de terceros (**obligatorio leer antes de uso comercial**).
