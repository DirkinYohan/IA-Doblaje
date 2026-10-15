# PIPELINE.md
## Pipeline de Análisis de Audio — 14 Pasos (Fase 1)

> Implementación progresiva. Steps T04+.

---

## Paso 01 — MEDIA VALIDATION
- Validar extensión whitelist
- Tamaño máximo (`MAX_MEDIA_SIZE_GB`)
- MIME / header sniffing
- SHA-256 del archivo (reproducibilidad)
- Safe path (no path-traversal)

## Paso 02 — AUDIO EXTRACTION
- FFmpeg via `subprocess.run([...], shell=False)` (seguro)
- Salida: WAV PCM 16-bit, 16 kHz, mono
- Timeout configurable
- Log: duración detectada vs duración extraída (sanity check)

## Paso 03 — AUDIO PREPROCESSING
- Resample (si fuera necesario)
- DC-offset removal
- Peak normalization
- High-pass filter 80 Hz (elimina ruido de infrasonido)

## Paso 04 — VOICE ACTIVITY DETECTION
- Model: **Silero VAD v5.1** (PyTorch JIT)
- Output: `List[VoiceInterval]` + `List[SilenceSegment]`

## Paso 05 — LANGUAGE DETECTION
- Whisper encoder → probabilidades multiclase
- Top-3 alternativas
- Umbral mínimo: 0.4 (fallback: user-specified o raise)

## Paso 06 — SPEECH-TO-TEXT
- Adapter desacoplado (`ASRModel` ABC)
  - **PERF**  → Faster-Whisper (CTranslate2) int8_float16
  - **BAL**   → Whisper-PyTorch medium FP16
  - **QLT**   → Whisper-PyTorch large-v3 FP16 beam=5

## Paso 07 — TIMESTAMPS GENERATION
- Segment timestamps (obligatorio)
- Word timestamps (si modelo lo permite)
- Post-validación: `start < end`, orden cronológico

## Paso 08 — SPEAKER DIARIZATION
- **pyannote.audio 3.x** (MIT + HF gated)
- SOLO output: `SPEAKER_00`, `SPEAKER_01`, ...
- **NO** identidad personaje. **NO** Speaker Clustering.
- ABC `SpeakerIdentityModel` reservado para Fase 5.

## Paso 09 — ASR ↔ SPEAKER ALIGNMENT
- Algoritmo: **Weighted Overlap Majority Vote**
- Tie-break: diarization confidence

## Paso 10 — QUALITY ANALYSIS
→ Ver [QUALITY_RULES.md](QUALITY_RULES.md). 11 reglas + score 0.0-1.0.

## Paso 11 — VALIDATION
- Schema Pydantic strict
- Checks de coherencia final
- Si Quality=FAILED y strict=true → aborta a menos `--force-save`

## Paso 12 — METRICS AGGREGATION
→ Ver [METRICS.md](METRICS.md).
- RTF (Real Time Factor)
- Peak CPU RSS / Peak GPU VRAM
- Utilizations, counts, avg confidences, etc.

## Paso 13 — STRUCTURED JSON OUTPUT
5 archivos:
1. `analysis.json`  → agregado completo
2. `transcript.json` → solo palabras + texto + timestamps
3. `speakers.json`   → SpeakerTurns raw (sin identidad)
4. `segments.json`   → DialogueSegments alineados
5. `metadata.json`   → media + processing + metrics

## Paso 14 — TEMP CLEANUP
- Auto-cleanup (UUID isolated dirs)

## Paso 15 — TRANSLATION ENGINE
- Motor: **facebook/m2m100_418M** (local, offline, many-to-many)
- Idiomas oficiales: `es, en, fr, de, it, pt, ja, zh`
- Entrada: `segments.json` + `transcript.json` (idioma origen)
- Flujo: traducción contextual → traducción natural → adaptación para doblaje → validación → `translation.json`
- Correspondencia **1:1** con `segments.json` (sin segmentos perdidos/duplicados)
- `source == target` → passthrough (sin ejecutar el modelo)
- `source == "und"` → error contractual determinista
- Adaptación: `max_chars_per_second=18.0`, `max_adaptation_ratio=1.35` (nunca trunca significado)
- Determinismo: `do_sample=False`, `num_beams=1`
- Se ejecuta tras T13 y antes de T14; T14 no elimina los outputs persistidos
- `DEV_KEEP_TEMP_FILES=true` para debugging

---

## Procesamiento progresivo (subtítulos mientras se procesa)

Además del pipeline completo T01→T15, existe un modo **progresivo** pensado
para que el usuario pueda reproducir el vídeo mientras la IA trabaja.

Flujo:

```text
SUBIR VÍDEO  ->  READY (no se procesa todavía)
      |
      v
"PRODUCIR"  ->  job en segundo plano
      |
      v
por ventanas de 30 s (solape 1 s): VAD -> ASR -> cues -> traducción -> DB
      |
      v
eventos en tiempo real (WebSocket /api/jobs/{id}/events)
      |
      v
subtítulos aparecen conforme se generan  ->  COMPLETED / PARTIAL
```

Diferencias con el pipeline completo:

| | Progresivo | Completo |
|---|---|---|
| Publica subtítulos | por ventana (30 s) | al terminar |
| Diarización | no (más rápido) | sí |
| Calidad/validación/métricas | no | sí |
| Eventos en vivo | `subtitle_created`, `processing_progress` | sólo progreso |
| Uso | reproducción inmediata | análisis profundo |

Puntos clave de la implementación:

* El audio original **no se modifica**: se extrae un WAV temporal a 16 kHz mono.
* El worker es independiente del reproductor: pausar el vídeo no lo detiene.
* El estado real está en SQLite; el WebSocket sólo evita el sondeo.
* Un fallo de traducción deja la pista vacía pero marca el trabajo como
  `PARTIAL` con la causa, nunca como éxito total.
