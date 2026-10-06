from app.application.use_cases.analyze_quality import RunQualityAnalysisUseCase
from app.domain.entities.asr import ASRSegment, ASRWord
from app.domain.value_objects.quality import QualityThresholds


class _Asr:
    def __init__(self, segments):
        self.segments = segments


def test_qr07_penaliza_segmento_bajo():
    seg = ASRSegment(segment_index=0, text="t", avg_logprob=-5.0, confidence=0.1)
    rule = RunQualityAnalysisUseCase._qr07(_Asr((seg,)), QualityThresholds())
    assert rule.applicable is True
    assert rule.passed is False
    assert rule.occurrences == 1


def test_qr08_penaliza_palabra_baja():
    word = ASRWord(text="hola", start_ms=0, end_ms=200, confidence=0.1)
    seg = ASRSegment(segment_index=0, text="hola", confidence=0.9, words=(word,))
    rule = RunQualityAnalysisUseCase._qr08(_Asr((seg,)), QualityThresholds())
    assert rule.applicable is True
    assert rule.occurrences == 1


def test_qr07_sin_dato_sigue_no_aplicable():
    seg = ASRSegment(segment_index=0, text="t")
    rule = RunQualityAnalysisUseCase._qr07(_Asr((seg,)), QualityThresholds())
    assert rule.applicable is False
    assert rule.reason == "segment_level_asr_confidence_unavailable"
