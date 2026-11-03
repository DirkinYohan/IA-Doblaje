from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.value_objects.translation import TranslationThresholds

_QUOTES_RE = re.compile(r"[\u2018\u2019\u201A\u201B\u2032\u2035]")
_DQUOTES_RE = re.compile(r"[\u201C\u201D\u201E\u201F\u2033\u2036]")
_DASH_RE = re.compile(r"[\u2013\u2014]")
_MULTISPACE_RE = re.compile(r" {2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

_FILLERS: dict[str, tuple[str, ...]] = {
    "es": ("bueno, ", "pues, ", "o sea, ", "digamos, ", "en plan, "),
    "en": ("well, ", "you know, ", "i mean, ", "actually, ", "like, "),
    "fr": ("eh bien, ", "ben, ", "en fait, ", "tu vois, "),
    "de": ("also, ", "na ja, ", "weißt du, ", "eigentlich, "),
    "it": ("beh, ", "insomma, ", "cioè, ", "praticamente, "),
    "pt": ("bem, ", "então, ", "tipo, ", "sabe, "),
    "ja": (),
    "zh": (),
}


@dataclass(frozen=True)
class AdaptationResult:
    adapted_text: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    chars_per_second: float = 0.0
    adaptation_ratio: float = 0.0
    fits_duration: bool = True


class TranslationAdaptationRule:
    @staticmethod
    def _normalize(text: str) -> str:
        s = text
        s = _QUOTES_RE.sub("'", s)
        s = _DQUOTES_RE.sub('"', s)
        s = _DASH_RE.sub("-", s)
        s = _MULTISPACE_RE.sub(" ", s)
        s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
        return s.strip()

    @staticmethod
    def _conservative_reduction(text: str, lang: str) -> str:
        s = text
        fillers = _FILLERS.get(lang, ())
        changed = True
        while changed:
            changed = False
            for f in fillers:
                if s.lower().startswith(f):
                    cand = s[len(f):].lstrip()
                    if cand:
                        s = cand
                        changed = True
        return s

    @staticmethod
    def _syntactic_simplification(text: str) -> str:
        s = text
        s = re.sub(r"\b(muy|really|très|sehr|molto|muito)\s+\1\b", r"\1", s)
        s = re.sub(r"\band also\b", "and", s)
        s = re.sub(r"\by también\b", "y", s)
        return s

    def adapt(
        self,
        *,
        translated_text: str,
        source_text: str,
        duration_ms: int,
        target_language: str,
        thresholds: TranslationThresholds | None = None,
    ) -> AdaptationResult:
        thr = thresholds if thresholds is not None else TranslationThresholds()
        warnings: list[str] = []

        base = str(translated_text).strip()

        adapted = self._normalize(base)

        adapted = self._conservative_reduction(adapted, target_language)

        adapted = self._syntactic_simplification(adapted)

        duration_sec = max(1e-6, int(duration_ms) / 1000.0)
        chars = len(adapted)
        cps = chars / duration_sec
        ratio = (chars / max(1, len(str(source_text).strip()))) if str(source_text).strip() else 0.0

        max_chars = int(thr.max_chars_per_second * duration_sec)
        fits = chars <= max_chars
        ratio_ok = ratio <= thr.max_adaptation_ratio

        if not fits:
            warnings.append(
                f"duración insuficiente: {chars} caracteres para {duration_ms}ms "
                f"(max {max_chars} @ {thr.max_chars_per_second} cps). Se conserva "
                f"la mejor versión válida sin truncar significado."
            )
        if not ratio_ok:
            warnings.append(
                f"ratio de expansión alto: {ratio:.2f} > {thr.max_adaptation_ratio} "
                f"(source={len(str(source_text).strip())} -> adapted={chars})."
            )

        return AdaptationResult(
            adapted_text=adapted,
            warnings=tuple(warnings),
            chars_per_second=round(cps, 3),
            adaptation_ratio=round(ratio, 3),
            fits_duration=fits,
        )


__all__ = ["AdaptationResult", "TranslationAdaptationRule"]
