# LICENSES_ANALYSIS.md
## Análisis Detallado de Cumplimiento Legal / Restricciones

> ⚠️ ESTADO: ESQUELETO T01. Contenido completo en T14.
> **Condición 1:** Nada aquí es consejo legal. Todo componente con
> incertidumbre se marca y debe ser revisado por abogado especializado.

---

## 1. METODOLOGÍA DE EVALUACIÓN (Condición 1)

Para cada componente, se revisan **fuentes oficiales actualizadas**:
1. Archivo `LICENSE` en el release tag exacto de GitHub/GitLab
2. `pyproject.toml` / `setup.py` classifier `License ::`
3. Hugging Face Model Card → sección License
4. TOS de la plataforma (HF Hub, NGC)
5. EULA / Legal FAQ del vendor
6. CHANGELOG para detectar cambios de licencia entre versiones

### Criterio de marca REQUIERE REVISIÓN LEGAL
Se marca cuando se cumple UNO de los siguientes:
- Build flags afectan la licencia (FFmpeg LGPL ↔ GPL)
- TOS / gate HF con texto ambiguo sobre uso comercial
- Licencias híbridas code MIT vs pesos CC
- Redistribución binaria de componentes propiedad de NVIDIA (CUDA/cuDNN)
- Dual licensing (AGPL/comercial, LGPL con cláusula static-link)
- Dependencias transitivas sin licencia SPDX confirmada

---

## 2. ANÁLISIS POR COMPONENTE CRÍTICO

### 2.1 PYTORCH
- **Riesgo:** 🟢 NINGUNO (BSD-3)
- **Comentario:** Librería principal. Redistribución completamente permitida.
- **Fuentes a verificar en T14:**
  - https://github.com/pytorch/pytorch/blob/v2.3.1/LICENSE
  - https://github.com/pytorch/audio/blob/v2.3.1/LICENSE

### 2.2 FFMPEG (BUILD LGPL) — ALTO IMPACTO
- **Riesgo:** 🟡 MEDIO. Licencia **depende críticamente de los flags de build**.
- **Escenario Incorrecto (❌ PROHIBIDO):**
  - Build con `--enable-gpl` + libx264 → licencia global pasa a **GPL-2.0-or-later**.
  - Si el producto se distribuye (SaaS / on-prem), **tienes que publicar TODO el código fuente** de toda la app que linkea FFmpeg. **Esto rompe el modelo propietario.**
  - Build con `--enable-nonfree` + libfdk-aac → ❌ NO redistribuir, NO uso comercial sin licencia Fraunhofer.
- **Escenario Correcto (✅ NUESTRO DEFAULT):**
  - Compilar desde fuente con flags:
    ```bash
    ./configure \
      --disable-gpl \
      --disable-nonfree \
      --enable-version3 \
      --enable-shared \
      --disable-static \
      --disable-programs \
      --enable-avcodec --enable-avformat --enable-swresample --enable-swscale \
      --enable-demuxer=mov,matroska,mp3,wav --enable-decoder=aac*,mp3*,pcm_s16le
    ```
  - Resultado: Licencia **LGPL-2.1-or-later (solo)**.
- **Obligaciones LGPL (cumplimiento):**
  1. Incluir texto licencia LGPL + atribución FFmpeg en el producto.
  2. Hacer disponible el script exacto de build (lo ponemos en `scripts/ffmpeg_build_lgpl.sh`).
  3. Dynamic link (shared libraries) → ya no obligación de publicar nuestro código.
- **Fuentes T14:**
  - https://ffmpeg.org/legal.html
  - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html

### 2.3 CUDA / CUDNN — REDISTRIBUCIÓN
- **Riesgo:** 🟡 MEDIO (por confusión static vs dynamic link).
- **Regla de oro adoptada:**
  - ❌ NO distribuimos `cudart64_110.dll`, `cudnn64_8.dll` ni ningún binario NVIDIA.
  - ✅ Usuario final instala driver NVIDIA + CUDA Toolkit. Docker image usa base oficial `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime` (NVIDIA ya licencia eso).
  - ✅ Dynamic linking únicamente.
- **Fuentes T14:**
  - https://docs.nvidia.com/cuda/eula/index.html
  - https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html

### 2.4 OPENAI WHISPER (PESOS large-v3)
- **Riesgo:** 🟢 BAJO (MIT confirmado OpenAI).
- **Historial a verificar (T14):**
  - Whisper v1 original: MIT.
  - Whisper v3 (nov 2023): ¿mantiene MIT? La evidencia apunta a sí (commit release notes), pero hay que confirmar en model card actual.
- **Fuentes T14:**
  - https://github.com/openai/whisper/blob/main/LICENSE
  - https://huggingface.co/openai/whisper-large-v3/blob/main/LICENSE

### 2.5 PYANNOTE.AUDIO v3 + GATED WEIGHTS
- **Riesgo:** 🟡 MEDIO + **🔍 REQUIERE REVISIÓN LEGAL** (por HF gate).
- **Historial de licencias pyannote (⚠️ TRAMPA):**
  | Versión | Licencia código | Licencia pesos | ¿Comercial? |
  |---|---|---|---|
  | 2.x (ANTIGUA, ❌ NO USAR) | MIT | ❌ CC-BY-NC-SA 4.0 | ❌ NO |
  | 3.x (ACTUAL, ✅ NUESTRA) | MIT | MIT model cards | ✅ PROBABLEMENTE SÍ, **pero** hay HF Gate TOS a revisar |
- **Qué es el HF Gate exactamente:**
  - Botón "Access repository" en: https://huggingface.co/pyannote/speaker-diarization-3.1
  - El usuario acepta:
    1. HF TOS (términos plataforma)
    2. TOS del proveedor del modelo (pyannote)
  - Muy a menudo el TOS del proveedor es "uso libre" pero hay que LEERLO.
- **Plan si el Gate TOS contiene cláusula NC/redist prohibida:**
  1. Feature flag `DIARIZATION_ADAPTER=speechbrain`
  2. Implementar adapter SpeechBrain diarization (Apache 2.0 completo, no gates)
  3. Trade-off: 2-5% peor DER, licencia 100% segura
- **Fuentes T14:**
  - https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE (MIT ✅)
  - HF Gate TOS real (texto del modal al aceptar access) — screenshot / TOS link.

### 2.6 SILERO VAD v5.1
- **Riesgo:** 🟢 BAJO (MIT código + pesos JIT).
- **Comentario:** Silero Team ha confirmado en múltiples issues que los pesos torchscript son MIT. Ver issue exacto + link en T14.

### 2.7 TRANSFORMERS / HUGGING FACE HUB
- **Riesgo:** 🟢 BAJO. HF code = Apache 2.0.
- **Aclaración:** La licencia de `transformers` NO implica nada sobre los pesos que descargas. Cada peso tiene su propia licencia → lo controlamos en tabla THIRD_PARTY_LICENSES sección 4/5/6.

---

## 3. MATRIZ DE CUMPLIMIENTO FINAL (PRE-T14)

| Escenario de Despliegue | Cumplimiento proyectado | Requisitos para pasar |
|---|---|---|
| 🖥️ **Desarrollo local (solo ingeniería)** | ✅ OK (sin distribución) | N/A |
| 🐳 **Imagen Docker interna (privada)** | ✅ OK si base `pytorch/*` oficial + build FFmpeg LGPL | Verificar Dockerfile no incluye CUDA .so extra |
| 🌐 **SaaS (API HTTP)** | ⚠️ CONDICIONAL | FFmpeg LGPL: SaaS/LGPL → se aconseja dynamic link ✔️. HF gated: 1 cuenta empresa acepta los gates. CUDA/cuDNN: runtime instalado servidor, no distribuido. |
| 💿 **On-Prem / Distribución binaria a cliente** | 🔴 REQUIERE REVISIÓN LEGAL | Obligaciones LGPL attribution + fuente build. Atención EULA NVIDIA re: redistribution. |
| 📱 **Embedded / Edge devices** | 🔴 REQUIERE REVISIÓN LEGAL | Static linking LGPL = publicar código. Dynamic link obligatorio. |

---

## 4. CHECKLIST PRE-COMERCIALIZACIÓN (OBLIGATORIO ANTES DE LAUNCH)

- [ ] Ejecutar `scripts/verify_third_party.py`. Todo link funciona, fechas < 90 días.
- [ ] ✅ Build FFmpeg **LGPL-only** reproducible. Hash binario + build script firmados.
- [ ] Todo componente marcado `REQUIERE REVISIÓN LEGAL` tiene **informe firmado por abogado**.
- [ ] Páginas "Legal / Third-Party Notices" en eventual producto web/app con attributions.
- [ ] Archivos licencia MIT/Apache/BSD/LGPL empaquetados en distribución final.
- [ ] Política de actualización: cada upgrade de dependencia = re-verify tabla licencias.
- [ ] CUDA/cuDNN: confirmado dynamic-link, no se redistribuyen DLLs NVIDIA.
- [ ] HF Token secreto NO hardcodeado en imágenes, distribuido via secrets manager.

---

## 5. REFERENCIAS (ESTADO T01, se amplían en T14)

1. **FSF/LGPL FAQ:** https://www.gnu.org/licenses/gpl-faq.html
2. **FFmpeg Legal:** https://ffmpeg.org/legal.html
3. **NVIDIA CUDA EULA:** https://docs.nvidia.com/cuda/eula/index.html
4. **Hugging Face TOS Models:** https://huggingface.co/terms
5. **Open Source Initiative Licenses:** https://opensource.org/licenses
6. **SPDX License List:** https://spdx.org/licenses/
7. **GitHub choosealicense:** https://choosealicense.com
