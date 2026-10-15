# THIRD_PARTY_LICENSES.md
## Registro de Cumplimiento de Licencias de Terceros — Fase 1

> ⚠️ ESTADO: ESQUELETO INICIAL (T01).
> El contenido completo de esta tabla (columnas "Verificación" y links)
> se completa EN T14 junto al script `scripts/verify_third_party.py`.
>
> **NORMATIVA DE ESTE DOCUMENTO (Condición 1):**
> Ninguna fila puede marcarse "Comercialmente Segura" sin verificación manual
> de las fuentes oficiales en la versión exacta del paquete.
> Cualquier duda razonable se marca como REQUIERE REVISIÓN LEGAL.

---

## CÓMO LEER ESTA TABLA

| Columna | Descripción |
|---|---|
| **Categoría** | Framework / Runtime / IA / Multimedia / Utilidades |
| **Componente** | Nombre y versión exacta (mismas que pyproject.toml) |
| **Uso en el proyecto** | Propósito concreto |
| **Licencia Código Fuente** | Del código fuente de la librería |
| **Licencia Pesos / Checkpoints** | De los weights / modelos preentrenados (N/A si no aplica) |
| **Términos del Modelo / TOS** | Condiciones específicas del modelo (HF gated, NVIDIA OML, etc.) |
| **Términos Plataforma** | TOS HuggingFace, PyPI, NVIDIA NGC, etc. |
| **Atribución Requerida** | Sí / No + qué tipo (texto, licencia incluida, fuente) |
| **Redistribución Permitida** | Binarios / Fuentes / Pesos — Condiciones |
| **Uso Comercial** | ✅ SI / ⚠️ CONDICIONAL / ❌ NO / 🔍 REQUIERE REVISIÓN LEGAL |
| **Verificación** | Fecha de última comprobación + links fuentes oficiales |
| **Notas / Restricciones** | Advertencias especiales |

---

## 1. CORE FRAMEWORK / RUNTIME

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **Python 3.11.x** | Runtime | PSF License v2 | N/A | N/A | python.org TOS | ✅ Sí (PSF) | ✅ Libre | ✅ SI | 🔍 Pendiente | |
| **PyTorch 2.3.x** | Framework IA central | BSD-3-Clause (Meta + Contribs) | N/A | N/A | pytorch.org EULA | ✅ Sí (BSD) | ✅ Libre | ✅ SI | 🔍 Pendiente | TorchVision? N/A esta Fase. |
| **torchaudio 2.3.x** | I/O + Transforms audio | BSD-2-Clause (Meta) | N/A | N/A | HF / PyPI | ✅ Sí | ✅ Libre | ✅ SI | 🔍 Pendiente | |
| **NumPy 1.26.x** | Álgebra lineal | BSD-3-Clause | N/A | N/A | SciPy TOS | ✅ Sí | ✅ Libre | ✅ SI | 🔍 Pendiente | |

---

## 2. NVIDIA CUDA / ACELERACIÓN GPU

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **CUDA Toolkit 11.8+** | Runtime + Compilación | NVIDIA CUDA EULA v11.x | N/A | N/A | NVIDIA EULA CUDA | ✅ Aceptar EULA usuario final | 🟡 DYNAMIC LINK SOLAMENTE. NO redistribuir binarios CUDA | ⚠️ CONDICIONAL | 🔍 **REQUIERE REVISIÓN LEGAL** | ❗ NO incluir CUDA .dll/.so en paquete distribuible. Dynamic link contra driver del sistema. |
| **cuDNN 8.9.x** | Kernels DNN | NVIDIA cuDNN EULA | N/A | N/A | NVIDIA EULA cuDNN | ✅ Sí | 🟡 Dynamic link. NO redistribuir. | ⚠️ CONDICIONAL | 🔍 **REQUIERE REVISIÓN LEGAL** | Idem CUDA. Build Docker incluye solo runtime NVIDIA oficial. |

---

## 3. MULTIMEDIA / FFmpeg

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **FFmpeg 6.0 (BUILD LGPL-ONLY)** | Extracción audio normalizada | **LGPL-2.1-or-later** (build sin módulos GPL) | N/A | N/A | FFmpeg Legal FAQ | ✅ ✅ (Atribución LGPL + fuente disponible) | ✅ SI Build LGPL. ❌ Build con x264/x265/fdk-aac. | ⚠️ CONDICIONAL | 🔍 **REQUIERE REVISIÓN LEGAL** | ❗ CRÍTICO: Build debe ejecutarse con `--disable-gpl --disable-nonfree`. Dockerfile oficial FASE-1 incluye build instructions. Licencia cambia radicalmente con flags equivocados. |

---

## 4. MODELOS IA: ASR

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **faster-whisper 1.0.x** | Adapter ASR perfil PERFORMANCE | MIT (Guillaume Klein / Systran) | N/A (pesos derivados de OpenAI) | N/A | HF Hub | ✅ Sí (MIT) | ✅ Libre | ✅ SI | 🔍 Pendiente | **Nota técnica (PUNTO 1):** Backend inferencia = **CTranslate2** (no PyTorch). Adapter alternativo Whisper-PyTorch usa PyTorch 100%. |
| **CTranslate2 4.x** | Motor inferencia (dep faster-whisper) | MIT (OpenNMT) | N/A | N/A | GitHub | ✅ Sí | ✅ Libre | ✅ SI | 🔍 Pendiente | |
| **openai/whisper-large-v3 (PESOS)** | Pesos ASR (ambos adapters) | N/A | **MIT (OpenAI release 2023-11)** | Whisper Model Card (MIT) | HF Hub + OpenAI TOS modelo | ✅ Atribución OpenAI Whisper | ✅ Libre (pesos MIT) | ✅ SI | 🔍 Pendiente | Verificar en HF model card URL oficial: https://huggingface.co/openai/whisper-large-v3/blob/main/LICENSE |
| **transformers 4.41.x** | Adapter Whisper-PyTorch (BAL + QLT) | Apache-2.0 (HuggingFace) | N/A | N/A | HF TOS | ✅ Apache NOTICE incluido | ✅ Libre | ✅ SI | 🔍 Pendiente | |
| **accelerate** | Inference PyTorch | Apache-2.0 | N/A | N/A | | ✅ | ✅ | ✅ SI | 🔍 Pendiente | |

---

## 5. MODELOS IA: VAD

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **snakers4/silero-vad v5.1** | Voice Activity Detection | MIT (Silero Team) | **MIT (TorchScript JIT weights)** | Silero Model Card | GitHub + HF Hub | ✅ Sí | ✅ Libre | ✅ SI | 🔍 Pendiente | Ver en repo oficial licencia confirmada para pesos. |

---

## 6. MODELOS IA: DIARIZATION (Fase 1 PUNTO 3)

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **pyannote.audio 3.3.x** | Library diarization | **MIT (v3.x confirmed)** | N/A | N/A | HF Hub | ✅ Sí | ✅ Libre | ✅ SI | 🔍 Pendiente | ❗ v2.x = ❌ CC-BY-NC-SA 4.0 NO COMERCIAL. FIJAR VERSIÓN: `pyannote.audio>=3.3,<3.4` excluye v2.x. |
| **pyannote/segmentation-3.0 (PESOS)** | Segmentation speaker embedding | N/A | **MIT (pesos card v3)** | ⚠️ **HF GATED: Aceptar model card 1 clic "pyannote TOS"** | HF Hub + pyannote TOS | ✅ Sí + Gate aceptado | ⚠️ CONDICIONAL (aceptar gate antes de download) | ⚠️ CONDICIONAL | 🔍 **REQUIERE REVISIÓN LEGAL** | ❗ Confirmar que la pyannote TOS del gate no impone cláusulas NC/ND. Si sí → fallback a SpeechBrain diarization (no gated). |
| **pyannote/speaker-diarization-3.1 (PESOS)** | Pipeline diarization | N/A | MIT (pesos card v3) | ⚠️ HF GATED (igual anterior) | HF Hub | ✅ Sí + gate | ⚠️ CONDICIONAL | ⚠️ CONDICIONAL | 🔍 **REQUIERE REVISIÓN LEGAL** | Idem. |
| **SpeechBrain diarization (FUTURO FALLBACK)** | Plan B sin gates | Apache-2.0 | Apache-2.0 (pesos) | N/A | HF Hub | ✅ Apache NOTICE | ✅ Libre | ✅ SI | 🔍 Pendiente | Se implementa como adapter alternativo si pyannote gates son problemáticos. |

---

## 7. LANGUAGE DETECTION

| Componente / Versión | Uso | Licencia Código | Licencia Pesos | Términos Modelo | Términos Plataforma | Atrib. Req | Redist. | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| **Whisper Encoder probs** | Detección idioma (reuse ASR) | MIT (OpenAI) | MIT (Whisper weights) | Idem ASR | Idem ASR | ✅ Sí | ✅ Libre | ✅ SI | 🔍 Pendiente | No descarga pesos extra. Reusa encoder. |

---

## 8. UTILIDADES / API / CLI / LINTING

| Componente / Versión | Uso | Licencia Código | Uso Comercial | Verificación T14 |
|---|---|---|---|---|
| FastAPI 0.111.x | Framework API (skeleton T01) | MIT | ✅ SI | 🔍 Pendiente |
| Pydantic 2.x + Settings | Validación / Settings | MIT | ✅ SI | 🔍 Pendiente |
| Typer 0.12.x | CLI | MIT | ✅ SI | 🔍 Pendiente |
| Rich 13.x | UI CLI | MIT | ✅ SI | 🔍 Pendiente |
| structlog 24.x | Logging estructurado | Apache-2.0 | ✅ SI | 🔍 Pendiente |
| huggingface-hub 0.23.x | Descarga modelos | Apache-2.0 | ✅ SI | 🔍 Pendiente |
| soundfile 0.12.x | I/O WAV | BSD-3 + libsndfile LGPL (dynamic) | ⚠️ CONDICIONAL — libsndfile dynamic link | 🔍 Pendiente |
| python-magic | MIME detection | MIT | ✅ SI | 🔍 Pendiente |
| psutil | Process metrics | BSD-3 | ✅ SI | 🔍 Pendiente |
| PyYAML | Config YAML | MIT | ✅ SI | 🔍 Pendiente |
| pytest + plugins | Testing | MIT / BSD / MPL | ✅ SI (solo dev/test) | 🔍 Pendiente |
| ruff | Linting + Format | MIT | ✅ SI (solo dev) | 🔍 Pendiente |
| mypy | Static type checker | MIT | ✅ SI (solo dev) | 🔍 Pendiente |
| bandit | SAST security | Apache-2.0 | ✅ SI (solo dev) | 🔍 Pendiente |
| pip-licenses | Verificación licencias | MIT | ✅ SI (solo dev) | 🔍 Pendiente |

---

## 9. PLATAFORMA / TOS HUGGING FACE

| Plataforma | Aplicación | Términos | Uso Comercial | Verificación T14 | Notas |
|---|---|---|---|---|---|
| **Hugging Face Hub TOS** | Descarga de todos los pesos de modelos (ASR, VAD, Diarization) | HF TOS sección "Datasets y Models" | ⚠️ CONDICIONAL: Algunos modelos tienen TOS propios (pyannote gated). Pesos Whisper = MIT sin TOS extra. | 🔍 **REQUIERE REVISIÓN LEGAL** | ❗ HF es plataforma. Las restricciones vienen de cada modelo individual, no de HF en general. |
| **HF INFERENCE API** | ❌ NO USAMOS en Fase 1. Todos los modelos son LOCALES. | N/A | N/A | ✅ N/A | |

---

## 10. RESUMEN DE RIESGOS

| Nivel de Riesgo | Componentes | Requiere Acción Pre-Comercial |
|---|---|---|
| 🔴 ALTO | Ninguno (si build FFmpeg LGPL correcto + pyannote gate OK) | — |
| 🟡 MEDIO | **FFmpeg**, **CUDA/cuDNN redistribution**, **pyannote HF gated weights TOS** | ✅ Build FFmpeg script reproducible<br>✅ Dynamic linking CUDA/cuDNN<br>✅ Aceptar pyannote gates + revisar texto real del TOS |
| 🟢 BAJO | Python, PyTorch, Whisper, Silero, Utils (FastAPI/Typer/...) | Incluir archivos LICENSE + NOTICE en distribución final |

---

> 📋 **Siguiente paso:** En T14, ejecutar `scripts/verify_third_party.py` para generar
> la tabla con links oficiales, fechas de comprobación y confirmación de cada fila.
> Los campos marcados `🔍 Pendiente` se actualizan con la evidencia concreta.
> Los campos `🔍 REQUIERE REVISIÓN LEGAL` pasan por un profesional legal.
