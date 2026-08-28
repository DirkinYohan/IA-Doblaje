"""IA Doblaje Engine - Fase 1: Motor de Analisis de Audio.

Package raiz de la aplicacion.
En T01 solo exporta metadatos basicos y la version.

Arquitectura (Clean Architecture):
    - app.core:           Cross-cutting (config, logging, exceptions, device)
    - app.domain:         Entities, ValueObjects, ABC Interfaces (sin dependencias externas)
    - app.application:    UseCases, Services (Quality, Alignment, Metrics, ...)
    - app.infrastructure: Implementaciones concretas de Interfaces
    - app.presentation:   Entrypoints CLI + FastAPI skeleton
"""

from __future__ import annotations

__title__ = "ia-doblaje-engine"
__version__ = "0.1.0"
__phase__ = "Fase 1 - Analisis de Audio (Pre-Alpha)"
__status__ = "in-development"
__author__ = "IA Doblaje Engineering Team"

__all__ = [
    "__title__",
    "__version__",
    "__phase__",
    "__status__",
    "__author__",
]
