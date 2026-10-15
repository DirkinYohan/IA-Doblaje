# QUALITY_RULES.md
## Módulo Quality Analysis — 11 Reglas de Validación

Implementación: `app/application/services/quality/*` (T09).

---

| Rule ID | Descripción | Severidad | Penalización (score 0→1) |
|---|---|---|---|
| QR01 | `start >= end` en cualquier segment | ERROR | Score = 0.0 (status=failed) |
| QR02 | Texto vacío / whitespace-only | ERROR | Score = 0.0 |
| QR03 | Timestamps imposibles (negativos, > duración audio) | ERROR | Score = 0.0 |
| QR04 | Segmentos superpuestos incorrectos (>100ms interspeech solape) | ERROR | -0.05 / ocurrencia |
| QR05 | Diálogo sin speaker asignado | WARNING | -0.03 / ocurrencia |
| QR06 | Speaker detectado pero 0 segmentos alineados | WARNING | -0.02 |
| QR07 | Segment ASR confidence < 0.4 | WARNING | -0.04 / segmento |
| QR08 | Palabra confidence < 0.30 | WARNING | -0.005 / palabra |
| QR09 | Segmentos > 60s (problemas TTS futuro) | WARNING | -0.02 / caso |
| QR10 | Silencios inter-speech > 15s (posible VAD miss) | WARNING | -0.015 / caso |
| QR11 | Inconsistencia ASR↔Diar: ASR detecta voz en segmento Silencio | WARNING | -0.03 / caso |

---

## 1. Cálculo del Score
```
score = 1.0 - Σ(penalties)
clamped = max(0.0, min(1.0, score))
```

## 2. Estados
| Score Umbral | Status |
|---|---|
| ≥ 0.80 | `passed` |
| 0.30 - 0.79 | `warning` |
| < 0.30 | `failed` |

## 3. Política Abort / Guardar
- `QUALITY_STRICT_MODE=true` (defecto) + `status=failed` → no guarda JSON
- Salvar con errores: `--force-save` CLI flag
