# Auditoría final — IA-Doblaje

**Fecha:** 6 de octubre de 2026
**Alcance:** `IA-Doblaje` (motor Python + API FastAPI) y `../video_subtitles` (frontend Next.js).
**Método:** cada afirmación se verificó ejecutando código real; no se usaron resultados simulados.

---

## 1. Veredicto

| Capa | Antes | Ahora | Evidencia |
|---|---|---|---|
| Modelos de IA | 2 de 5 presentes | **5 de 5 cargando** | Carga real por adapter |
| Pipeline | No ejecutable | **`status=SUCCESS`** | 15 pasos sobre vídeo real |
| API | 27 rutas `/api` | **29 rutas `/api`** (33 con las de documentación) | Preflight, PATCH perfil, miniatura |
| Tests motor | 1140 | **1214** | `pytest tests -q` |
| Tests frontend | 2 (uno tautológico) | **4 suites reales** | `npm test` |
| Flujo E2E | Roto | **Completo** | API + worker real |

El flujo **VÍDEO → IA → SUBTÍTULOS → TRADUCCIÓN → REPRODUCTOR → EXPORTACIÓN** funciona de extremo a extremo.

---

## 2. Modelos: los 3 ausentes están operativos

| Paso | Modelo | Tamaño | Estado |
|---|---|---|---|
| T04 VAD | `silero_vad_v5.1.jit` | 2,3 MB | ✅ Carga e infiere (`silero-vad:v5.1`) |
| T05 LID | `whisper_encoder_small_multilingual_lid_v1.jit` | 954,8 MB | ✅ Exportado y validado |
| T06 ASR | `whisper_small_ct2_int8_fp16_local_v1/model.bin` | 483,5 MB | ✅ |
| T08 Diarización | `pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin` | 26,6 MB | ✅ `WeSpeakerResNet34` |
| T15 Traducción | `m2m100_418M/pytorch_model.bin` | 1935,8 MB | ✅ |

**T05 se generó con el script existente**, sin inventar arquitectura:

```text
python scripts/export_whisper_lid_jit.py --checkpoint models/whisper_small_pytorch/small.pt
→ output shape: (1, 99) · VALIDACIÓN OUTPUT: PASS · 954.8 MB
```

Los demás se obtuvieron de sus fuentes oficiales (torch.hub para Silero, HF Hub para WeSpeaker, CDN de OpenAI para los pesos de Whisper).

---

## 3. Ejecución real del pipeline

Vídeo real de 222 s (`[B1] What Do You Do on Weekends.mp4`), ejecutado por el **worker real a través de la API**:

```text
status = SUCCESS                       (52-69 s según la corrida)
detected_language = 'en'               (confianza 0.998)
ASR segments = 69
VAD total_speech_ms = 187840 (84,5 % de voz)
Diarization = 148 turnos, 6 hablantes
Subtitle cues = 74 (origen) · 77 tras traducir
Traducción -> es: 68 segmentos
quality_score = 0.965 · validation_status = passed
```

### Artefactos verificados en contenido, no solo en existencia

| Comprobación | Resultado |
|---|---|
| SRT: 77 bloques, índices contiguos 1..77 | ✅ |
| SRT: sin solapes, `end > start`, ≤ 2 líneas | ✅ |
| VTT empieza por `WEBVTT`, timestamps `HH:MM:SS.mmm` | ✅ |
| ASS con `[Script Info]`, `[V4+ Styles]`, `[Events]`, `Dialogue:` | ✅ |
| ZIP con `en.srt`, `en.vtt`, `en.ass`, `es.srt`, `es.vtt`, `es.ass` | ✅ |
| UTF-8 real: `inglés`, `día`, `¿`, `¡` (0xE9, 0xED, 0xBF, 0xA1) | ✅ |
| Traducción real: 74/77 cues con texto distinto del original | ✅ |

---

## 4. Correcciones aplicadas

### 4.1 Bloqueadores

| # | Problema | Solución | Archivo |
|---|---|---|---|
| 1 | Faltaban 3 modelos | Obtenidos/exportados desde fuentes oficiales | `models/` |
| 2 | `pyannote.audio` no instalado | Instalado 4.0.7 | `pyproject.toml` |
| 3 | `umap`/`hdbscan` ausentes: el clustering de hablantes fallaba | Instalados y declarados | `pyproject.toml` |
| 4 | `torchaudio.load` pasa por `torchcodec`, que exige DLL de FFmpeg **compartidas**; el build de WinGet es estático → T08 fallaba siempre | Lectura de WAV por `soundfile` (dependencia ya declarada) | `app/infrastructure/audio/wav_io.py` |
| 5 | `silero_vad.read_audio` tenía la misma dependencia | Reutiliza el lector anterior | `wespeaker_diarization_adapter.py` |
| 6 | `.venv-doblaje` roto (apuntaba a un usuario inexistente) | Eliminado | — |

### 4.2 Idioma detectado

`worker.py` ahora hace `UPDATE projects SET source_language=?` con el resultado de T05.
El pipeline expone `results["detected_language"]`. Verificado: subir con `und` deja `source_language='en'`.

### 4.3 Progreso: monótono, con etapas reales

Antes: 4 etapas genéricas, el porcentaje **retrocedía** de 55 a 40, y un fallo en T04 se atribuía a `T01_T02_T03`.

Ahora existe un modelo canónico en `app/presentation/api/progress.py`:

```text
QUEUED 0 → PREPARING 3 → MEDIA_PREP 8 → VAD 16 → LID 22 → ASR 32
→ TIMESTAMPS 52 → DIARIZATION 58 → ALIGNMENT 66 → QUALITY 72
→ VALIDATION 76 → METRICS 79 → OUTPUT 82 → TRANSLATING 88
→ GENERATING_SUBTITLES 95 → COMPLETED 100
```

* `ProgressTracker` garantiza monotonía (nunca decrece).
* El pipeline anuncia el **inicio** de cada paso (`_notify(..., "start")`), así que la UI ve la etapa mientras se ejecuta.
* `T01/T02/T03` emiten subetapas para que la preparación no parezca congelada.
* **Historial persistido en servidor** (`jobs.stage_history`, JSON): ninguna etapa puede perderse aunque el sondeo llegue tarde.

Verificado en vivo: `[0, 3, 8, 22, 32, 58, 100]` observado por sondeo; `[0, 3, 8, 16, 22, 32, 52, 58, 66, 72, 76, 79, 82, 88, 95, 100]` persistido — **16/16 etapas**.

### 4.4 Subida por streaming

`await file.read()` (archivo completo en RAM) sustituido por lectura por bloques de 1 MiB escritos a disco, con:

* límite de tamaño aplicado **durante** la escritura,
* borrado del archivo parcial si se excede o falla,
* saneado del nombre (`../../evil name;rm.mp4` → nombre seguro),
* `probe` de duración con ffprobe y generación de miniatura.

### 4.5 Recuperación de jobs

`worker.reconcile_orphans()` se ejecuta al arrancar la API: todo job en estado no terminal se reencola. Un reinicio ya no deja `PROCESSING` eterno. Con el worker desactivado, el job se marca `FAILED` con causa explícita en vez de mentir.

### 4.6 API

| Cambio | Detalle |
|---|---|
| `PATCH /api/users/me` | Editar nombre (2–80 caracteres) |
| `GET /api/videos/{id}/thumbnail` | Miniatura JPEG con validación de ruta |
| `GET /api/system/preflight` | Estado real del motor |
| `GET /api/videos` | Ahora incluye `duration_ms` y `thumbnail_url` |
| `GET /api/jobs/{id}` y `/progress` | Incluyen `stage_history` |
| Preflight | Usa el mismo resolutor de FFmpeg que el pipeline (antes daba falso negativo porque `shutil.which` no ve WinGet) |
| `realtime` | `translations` siempre presente (contrato estable) |

### 4.7 Frontend

| Área | Antes | Ahora |
|---|---|---|
| Tarjetas | `<video>` oculto por tarjeta para leer la duración | `duration_ms` y miniatura desde la API |
| Polling | 2 intervalos sobre el mismo job (2 req/2 s) | Un único `useJobProgress` |
| Errores | 5 `fetch` sin `try/catch`; uno rechazaba dentro de un `setInterval` | Cliente con `ApiError` + `errorMessageKey` traducido |
| `onBind` | `useEffect` **sin dependencias**, rebindeo continuo | `seekRef` con dependencias correctas |
| Fuente de verdad | Editor, timeline y player con estado propio | `WatchStudio` es el único dueño |
| Timeline | Solo visualización + clic | Playhead, scrubbing con puntero, zoom (1×–16×), regla temporal, selección, teclado (←/→/Inicio/Fin) |
| Tiempo real | `ScriptProcessorNode` deprecado, siempre activo, sin control | `AudioWorklet` (`public/pcm-tap-worklet.js`), toggle, indicador de conexión, reconexión con backoff, cierre real, backpressure |
| Modal | Sin Escape, sin focus trap, sin bloqueo de scroll | Las tres cosas + devolución de foco |
| LanguageMenu | `role="listbox"` mal usado | `menu`/`menuitemradio` con navegación por flechas |
| Rutas | 5 redirects huérfanos | Eliminadas |
| Código muerto | `ProjectList.tsx` (103 líneas) y `Uploader.tsx` (128) sin usar | Eliminados; su búsqueda/filtros/orden se integraron en `ProjectBoard` |
| i18n | 53 claves muertas + 1 duplicada | 0 muertas, 0 duplicadas, 195 claves es/en |
| Colores | ~15 literales fuera de la paleta | Tokens en `@theme` (`line-strong`, `stage`, `skeleton`, `danger-softer`…) |
| Tests | `cues.test.mjs` **reimplementaba** la función | Importa y transpila `lib/cues.ts` real |

### 4.8 Entorno reproducible

`pyproject.toml` declaraba `numpy<2.1`, `torch<2.5`, `transformers<4.45`, `python<3.13`… mientras el entorno real usa numpy 2.5.3, torch 2.14.1, transformers 5.18.0, Python 3.14.7. Se optó por **(B) alinear las restricciones con lo verificado**, documentando las versiones probadas y abriendo los techos para no bloquear parches de seguridad. Extras `asr`, `diarization` y `vad` actualizados con lo que el motor realmente necesita.

### 4.9 Página en blanco en la home (corregido)

Al hacer accesible el `Modal` se añadió `"use client"` a `components/ui.tsx`. Eso convirtió `buttonClass()` en una función de cliente, y `app/page.tsx` —que es **Server Component**— la llamaba:

```text
⨯ Attempted to call buttonClass() from the server but buttonClass is on the client.
  at HomePage (app\page.tsx:21:58)
```

El HTML del servidor salía, pero la hidratación fallaba y la página quedaba en blanco.

**Solución:** las clases CSS se movieron a `components/classes.ts` (sin `"use client"`), que exporta sólo cadenas y funciones puras y por tanto puede importarse desde el servidor. `components/ui.tsx` las reexporta para los Client Components.

**Prevención:** `lib/boundaries.test.mjs` recorre los 43 módulos del frontend, detecta cualquier Server Component que **llame** a una función exportada por un módulo cliente, y verifica que `components/classes.ts` no lleve `"use client"`. Se comprobó que la guardia falla (exit 1) si se reintroduce el bug.

También se añadió `allowedDevOrigins: ["127.0.0.1", "localhost"]` en `next.config.ts`: la app se abre por `127.0.0.1` y Next bloqueaba sus recursos de desarrollo (HMR) como origen cruzado.

### 4.10 Metadatos de proyectos antiguos (corregido)

Los proyectos subidos **antes** de existir la miniatura y la duración tenían `duration_ms=0` y `thumbnail_path=""`. `_ensure_project_metadata()` los rellena de forma perezosa al listar o abrir el proyecto, una sola vez y sin poder romper la respuesta. Verificado: el proyecto antiguo pasó de `0 ms` / sin miniatura a `222272 ms` / JPEG real, y la segunda llamada tarda 0,02 s (ya está en base de datos).

---

## 5. Verificación final

| Comprobación | Resultado |
|---|---|
| `pytest tests -q` | **1214 passed** (15 s) |
| `npx tsc --noEmit` | **0 errores** |
| `npm run lint` | **0 errores, 0 warnings** |
| `npm test` | **4/4** (cues real, status, i18n-keys, i18n checker) |
| `npm run build` | **✓ Compiled successfully**, 10 rutas |
| E2E API + worker real | **COMPLETO** (10 bloques) |
| Preflight del motor | **ok=True**, 0 fallos |

### Cobertura de pruebas nueva

| Archivo | Qué prueba |
|---|---|
| `tests/presentation/test_api_progress.py` | 31 tests: catálogo de etapas, monotonía, mapeo de pasos |
| `tests/presentation/test_api_integration.py` | 24 tests: auth, perfil, subida streaming, miniatura, saneado, jobs, cancelación, aislamiento entre usuarios, roundtrip de subtítulos, export SRT/VTT/ASS/ZIP |
| `tests/presentation/test_api_realtime_ws.py` | 5 tests: WebSocket con PCM real, parcial/final, seek, resume, tramas inválidas |
| `tests/presentation/test_api_worker.py` | 14 tests: migración idempotente, reconciliación de huérfanos, historial |
| `lib/cues.test.mjs` | 30 aserciones sobre `lib/cues.ts` real |
| `lib/status.test.mjs` | Estados, etapas y errores que emite el backend |
| `scripts/check-i18n.mjs` | Claves muertas, duplicadas, faltantes y paridad es/en |

---

## 6. Limitaciones y trabajo pendiente

Nada de esto impide el flujo principal, pero conviene saberlo.

### 6.1 Verificación en navegador no realizada

El correcto funcionamiento del **AudioWorklet** y de la **timeline** en un navegador real no se pudo comprobar aquí: no hay navegador disponible en este entorno. Sí están verificados:

* el contrato del WebSocket contra el servidor real (parcial → final → traducción → seek → resume),
* el `tsc`, `eslint` y `build` del código del cliente,
* que `pcm-tap-worklet.js` se sirve desde `public/` y el hook lo registra con `audioWorklet.addModule`.

Falta la prueba manual: abrir un proyecto, activar «En vivo» y confirmar que aparecen subtítulos provisionales mientras se reproduce.

### 6.2 Rendimiento

* En CPU, un vídeo de 222 s tarda **~52-69 s** (RTF ≈ 3,1 en el tramo de diarización, que es el más costoso).
* La diarización ejecuta su **propio VAD interno** además del T04. Es redundante (T04 ya calcula 53 intervalos que T08 no reutiliza) y para vídeos largos duplicaría ese coste.
* `M2M100` traduce segmento a segmento sin lote pese a existir `translate_batch`.

### 6.3 Otros

* `onnxruntime` está instalado pero no lo usa ningún adapter (el VAD va por TorchScript).
* Las restricciones de `pyproject.toml` ya reflejan el entorno, pero **no hay lockfile** (`uv.lock`/`requirements.txt` con hashes).
* El frontend no tiene pruebas de componentes (no hay React Testing Library ni Vitest).
* `data/output/<job_id>/` conserva los resultados de cada job: no hay política de retención.

---

## 7. Cómo reproducir la verificación

```powershell
# Motor + API
cd C:\Users\dirki\Documents\IA\IA-Doblaje
.venv\Scripts\python.exe -m pytest tests -q

# Preflight del motor (modelos, binarios, directorios)
.venv\Scripts\python.exe -c "from app.application.pipeline.preflight import run_preflight; [print(('OK  ' if i.ok else 'FALLA'), i.component, i.detail) for i in run_preflight()]"

# API + worker reales
.venv\Scripts\python.exe -m app.presentation.api

# Frontend
cd ..\video_subtitles
npm run verify      # typecheck + lint + tests + build
```
