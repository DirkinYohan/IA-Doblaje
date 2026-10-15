# ARCHITECTURE.md
## Arquitectura del Sistema — IA Doblaje Engine (Fase 1)

> ⚠️ Documento EN CONSTRUCCIÓN. Versión completa en T02-T03.

---

## 1. Principios de Diseño
- **Clean Architecture** (Uncle Bob) adaptada a ML/IA pipelines
- **SOLID**, DRY, KISS
- **Dependency Inversion**: toda integración con terceros (modelos IA, FFmpeg, storage)
  se hace a través de ABC (Abstract Base Classes) definidos en `app/domain/interfaces/`
- **Sin variables globales**: todo se pasa via Settings / Dependency Injection manual
- **Offline-first preparado**: modelos locales, caché HF en `models/`

---

## 2. Capas (de adentro hacia afuera)
```
DOMAIN LAYER       → Sin dependencias externas. Entidades + ValueObjects + ABCs
APPLICATION LAYER  → UseCases + Services (dependen solo de Domain)
INFRASTRUCTURE LAYER → Implementaciones concretas de los ABC de Domain
PRESENTATION LAYER → CLI (Typer) + FastAPI Skeleton
```

### 2.1 Dependencia Estricta
- **Domain** NO puede importar de Application/Infrastructure.
- **Application** NO puede importar de Infrastructure (solo usa Domain ABCs).
- **Infrastructure** implementa los ABCs de Domain.
- **Presentation** llama a los UseCases de Application.

---

## 3. Diagrama de Flujo del Pipeline
→ Ver [PIPELINE.md](PIPELINE.md)

## 4. Separación Speaker → Identity → Character (PUNTO 3)
→ Ver diagrama en README sección Arquitectura.
Fase 1 implementa únicamente **Speaker Diarization** (SPEAKER_XX).

## 5. Perfiles Quality/Balanced/Performance (PUNTO 6)
→ Ver sección en README / `.env.example` / `configs/*.yaml` (T02).

## 6. Quality Analysis (PUNTO 4)
→ Ver [QUALITY_RULES.md](QUALITY_RULES.md) — 11 reglas QR01-QR11.

## 7. Métricas (PUNTO 7)
→ Ver [METRICS.md](METRICS.md).

## 8. Cumplimiento Legal / Licencias (PUNTO 1 Condición 1)
→ **OBLIGATORIO LEER ANTES DE CUALQUIER USO COMERCIAL:**
   - [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
   - [LICENSES_ANALYSIS.md](LICENSES_ANALYSIS.md)
