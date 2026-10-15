# METRICS.md
## Diccionario de Métricas (PUNTO 7)

Todas las métricas se almacenan en `analysis.json → metrics{}`
y se pueden visualizar con `python -m app metrics <analysis.json>` (T12+).

---

## 1. Tiempos / Rendimiento
| Campo | Unidad | Descripción |
|---|---|---|
| `processing_wall_time_sec` | s | Tiempo real total de ejecución (wall-clock) |
| `audio_duration_sec` | s | Duración del audio de entrada |
| **`real_time_factor_RTF`** | ratio | **KPI principal**: `processing / audio`. Ideal < 1.0 (mas rápido que tiempo real) |
| `step_timings_sec` | dict[str, float] | Tiempo por cada uno de los 14 PipelineSteps |

## 2. Memoria
| Campo | Unidad | Descripción |
|---|---|---|
| `peak_cpu_rss_mb` | MB | Peak RAM (Resident Set Size) medido via psutil |
| `peak_gpu_vram_used_mb` | MB | Peak VRAM usado (torch.cuda.max_memory_allocated) |
| `peak_gpu_vram_total_mb` | MB | VRAM total de la GPU |

## 3. Utilización (Promedio)
| Campo | % |
|---|---|
| `cpu_utilization_avg_percent` | 0-100 |
| `gpu_utilization_avg_percent` | 0-100 |

## 4. Conteos / Estadísticas Pipeline
| Campo |
|---|
| `segment_count_total` |
| `segment_count_with_text` |
| `word_count_total` |
| `word_count_confidence_ge_08` |
| `speaker_count_detected` |
| `silence_segment_count` |

## 5. Calidad / Confianza
| Campo | Rango |
|---|---|
| `average_asr_confidence` | 0-1 |
| `average_diarization_confidence` | 0-1 |
| `language_detection_confidence` | 0-1 |
| **`overall_quality_score`** | **0-1** (Quality Analysis, PUNTO 4) |

## 6. Compliance / Decisiones Usuario (PUNTO 2)
| Campo | Descripción |
|---|---|
| `profile_requested` | Perfil pedido por el usuario |
| `profile_applied` | Perfil efectivamente ejecutado |
| `profile_downgrade_applied` | **bool**: `true` si hubo downgrade explícito |
| `profile_downgrade_reason` | Razón documentada (ej: "VRAM insuficiente, user eligió BALANCED") |
| `device_requested` | |
| `device_used` | |
