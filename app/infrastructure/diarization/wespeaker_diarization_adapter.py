"""Infrastructure Adapter Step 08 — WeSpeaker Diarization (alternativa local).

OPCIÓN B autorizada (FASE C.5): diarización local real usando WeSpeaker
(checkpoint oficial `pyannote/wespeaker-voxceleb-resnet34-LM`, CC-BY-4.0) +
Silero VAD (ya presente en T04) + clustering oficial WeSpeaker
(UMAP + HDBSCAN + PAHC, Apache-2.0).

Flujo real (audio → segmentación/actividad → embeddings → clustering → labels):
    1. VAD (Silero) sobre el WAV para hallar intervalos de voz.
    2. Por cada intervalo de voz: embeddings por ventana deslizante
       (Inference window="sliding", duration=1.5 s, step=0.75 s) con el
       modelo WeSpeaker ResNet34-LM (pyannote.audio).
    3. Clustering UMAP + HDBSCAN + PAHC (algoritmo oficial de WeSpeaker).
    4. merge_segments → SpeakerTurn normalizados SPEAKER_NN.

Offline 100%: el checkpoint es local (models/pyannote/wespeaker-voxceleb-
resnet34-LM/pytorch_model.bin). Sin HF hub, sin URLs, sin tokens.

El adapter implementa SpeakerDiarizerPort y mapea la salida a
DiarizationResult / SpeakerTurn (contratos existentes). No importa ni
contamina el dominio con WeSpeaker.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.core.exceptions import DiarizationError, ModelLoadError
from app.domain.entities.diarization import (
    DIARIZATION_STRATEGY_PYANNOTE,
    MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
    DiarizationResult,
)
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.interfaces.diarization_ports import SpeakerDiarizerPort
from app.domain.value_objects.diarization import (
    DiarizationThresholds,
    SpeakerTurn,
)

# ---------------------------------------------------------------------------
# Constantes (rutas relativas dentro de models_cache_dir)
# ---------------------------------------------------------------------------
MODEL_FOLDERNAME = "pyannote"
WESPEAKER_SUBFOLDER = "wespeaker-voxceleb-resnet34-LM"
WESPEAKER_CHECKPOINT_FILENAME = "pytorch_model.bin"

# Parámetros de diarización (defaults oficiales WeSpeaker)
_DIAR_WINDOW_SECS = 1.5
_DIAR_PERIOD_SECS = 0.75
_DIAR_BATCH_SIZE = 32
_DIAR_MIN_DURATION_SEC = 0.255


# ---------------------------------------------------------------------------
# Loader local (inyectable para tests) — SIN descargas, SIN red.
# ---------------------------------------------------------------------------


class WespeakerDiarizationModelLoader:
    """Valida y carga el modelo de embeddings WeSpeaker localmente (offline)."""

    def __init__(self, settings_getter: Any | None = None) -> None:
        self._settings_getter = settings_getter

    def _get_models_dir(self) -> Path:
        if self._settings_getter is not None:
            settings = self._settings_getter()
            models_dir = settings.paths.models_cache_dir
        else:
            from app.core.config import get_settings

            models_dir = get_settings().paths.models_cache_dir
        return Path(models_dir).expanduser().resolve()

    def _checkpoint_path(self) -> Path:
        return (
            self._get_models_dir() / MODEL_FOLDERNAME / WESPEAKER_SUBFOLDER / WESPEAKER_CHECKPOINT_FILENAME
        )

    def validate(self) -> Path:
        """Valida que el checkpoint local exista y sea un archivo razonable."""
        cp = self._checkpoint_path()
        if not cp.is_file():
            raise ModelLoadError(
                f"WeSpeaker checkpoint no encontrado: {cp!s}. "
                "Colocar manualmente (regla T08: NO descargar automáticamente)."
            )
        if cp.stat().st_size < 1024:
            raise ModelLoadError(f"WeSpeaker checkpoint corrupto (size < 1KB): {cp!s}")
        return cp

    def load(self, checkpoint_path: Path) -> Any:
        """Carga el modelo pyannote WeSpeakerResNet34 desde checkpoint local."""
        try:
            from pyannote.audio import Model as _PyannoteModel
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"pyannote.audio no disponible: {exc!r}") from exc
        try:
            return _PyannoteModel.from_pretrained(str(checkpoint_path))
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"No se pudo cargar WeSpeaker desde {checkpoint_path!s}: {exc!r}"
            ) from exc


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class WespeakerSpeakerDiarizationAdapter(SpeakerDiarizerPort):
    """Diarización local WeSpeaker (ResNet34-LM + Silero VAD + clustering).

    Lazy singleton. Offline estricto. Mapea a DiarizationResult/SpeakerTurn.
    """

    def __init__(
        self,
        *,
        model_loader: Any | None = None,
        settings_getter: Any | None = None,
    ) -> None:
        self._loader: Any = model_loader or WespeakerDiarizationModelLoader(
            settings_getter=settings_getter
        )
        self._model: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    def _get_model(self) -> Any:
        if not self._loaded:
            cp = self._loader.validate()
            self._model = self._loader.load(cp)
            self._loaded = True
        return self._model

    # ------------------------------------------------------------------
    def diarize(
        self,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        *,
        thresholds: DiarizationThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> DiarizationResult:
        thr: DiarizationThresholds = thresholds or DiarizationThresholds()
        total_ms = int(round(float(preprocessed_audio.duration_sec) * 1000.0))
        t0 = time.perf_counter()

        model = self._get_model()

        try:
            turns_sec = self._run_diarization(
                str(preprocessed_audio.wav_path),
                int(preprocessed_audio.sample_rate),
                int(thr.max_num_speakers),
            )
        except Exception as exc:  # noqa: BLE001
            raise DiarizationError(
                f"WeSpeaker diarization falló sobre "
                f"{Path(preprocessed_audio.wav_path).name!r}: {exc!r}"
            ) from exc

        # Convertir [begin_sec, end_sec, label] -> SpeakerTurn (ms)
        speaker_turns: list[SpeakerTurn] = []
        for begin_s, end_s, label in turns_sec:
            s_ms = max(0, int(round(begin_s * 1000.0)))
            e_ms = min(total_ms, int(round(end_s * 1000.0)))
            if e_ms <= s_ms:
                continue
            speaker_turns.append(
                SpeakerTurn(
                    speaker_label=label,
                    start_ms=s_ms,
                    end_ms=e_ms,
                    confidence=1.0,
                )
            )

        # Ordenar ASC y resolver solapamientos (misma semántica que pyannote adapter)
        speaker_turns = self._resolve_overlaps(speaker_turns)
        # Duración mínima de turno
        speaker_turns = [
            t for t in speaker_turns
            if (t.end_ms - t.start_ms) >= int(thr.min_speaker_duration_ms)
        ]

        num_speakers = len({t.speaker_label for t in speaker_turns})

        if logger is not None:
            try:
                logger.info(
                    "wespeaker_diarization_done",
                    extra={
                        "elapsed_sec": round(time.perf_counter() - t0, 4),
                        "num_speakers": num_speakers,
                        "num_turns": len(speaker_turns),
                        "job_id": job_id,
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        return DiarizationResult(
            source_preprocessed_sha256=str(preprocessed_audio.sha256),
            vad_reference_sha256=str(vad_result.source_preprocessed_sha256),
            total_duration_ms=total_ms,
            speaker_turns=tuple(speaker_turns),
            num_speakers=num_speakers,
            num_turns=len(speaker_turns),
            strategy=DIARIZATION_STRATEGY_PYANNOTE,
            model_label=MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
            job_id=job_id,
            analysis_metadata={"source": "wespeaker.resnet34-lm.local.v1"},
        )

    # ------------------------------------------------------------------
    # Núcleo de diarización (testable sin red)
    # ------------------------------------------------------------------
    def _run_diarization(
        self, wav_path: str, sample_rate: int, max_speakers: int
    ) -> list[tuple[float, float, str]]:
        import numpy as np
        import torch
        import torchaudio

        # 1) VAD Silero sobre el WAV (activity/segmentation)
        from silero_vad import get_speech_timestamps, read_audio

        wav_for_vad = read_audio(wav_path)
        vad_segments = get_speech_timestamps(wav_for_vad, self._get_vad_model(), return_seconds=True)
        if not vad_segments:
            return []

        pcm, sr = torchaudio.load(wav_path, normalize=False)
        if sr != sample_rate:
            pcm = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(pcm)
        if pcm.size(0) > 1:
            pcm = pcm.mean(dim=0, keepdim=True)
        pcm = pcm.to(torch.float)

        from pyannote.audio import Inference

        inference = Inference(
            self._model,
            window="sliding",
            duration=_DIAR_WINDOW_SECS,
            step=_DIAR_PERIOD_SECS,
            batch_size=_DIAR_BATCH_SIZE,
        )

        subseg_spans: list[tuple[float, float]] = []
        embeddings_list: list[np.ndarray] = []

        for item in vad_segments:
            begin, end = float(item["start"]), float(item["end"])
            if end - begin < _DIAR_MIN_DURATION_SEC:
                continue
            begin_idx = max(0, int(begin * sample_rate))
            end_idx = min(pcm.shape[1], int(end * sample_rate))
            if end_idx <= begin_idx:
                continue
            seg_wav = pcm[:, begin_idx:end_idx]
            # embeddings por ventana deslizante sobre el segmento de voz
            feat = inference.slide(seg_wav, sample_rate, hook=None)
            import pyannote.core

            if isinstance(feat, pyannote.core.SlidingWindowFeature):
                embs = np.asarray(feat.data, dtype=np.float32)
            else:
                embs = np.asarray(feat, dtype=np.float32)
                if embs.ndim == 1:
                    embs = embs.reshape(1, -1)

            num_chunks = embs.shape[0]
            for c in range(num_chunks):
                start_s = begin + c * _DIAR_PERIOD_SECS
                end_s = min(end, start_s + _DIAR_WINDOW_SECS)
                subseg_spans.append((start_s, end_s))
                embeddings_list.append(embs[c])

        if not embeddings_list:
            return []

        embeddings = np.vstack(embeddings_list)

        # 2) Clustering oficial WeSpeaker
        labels = self._cluster(embeddings, max_speakers)

        # 3) subsegmentos -> [begin_s, end_s, label]
        subseg2label: list[tuple[float, float, int]] = []
        for span, label in zip(subseg_spans, labels):
            subseg2label.append((span[0], span[1], int(label)))

        # 4) merge_segments (oficial WeSpeaker)
        merged = self._merge_segments(subseg2label)

        # 5) Normalizar labels a SPEAKER_NN deterministas (por orden de aparición)
        return self._normalize_to_speaker_labels(merged)

    # ------------------------------------------------------------------
    # Helpers (puros, testables)
    # ------------------------------------------------------------------

    def _get_vad_model(self) -> Any:
        if not hasattr(self, "_vad"):
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad()
        return self._vad

    @staticmethod
    def _cluster(embeddings: Any, max_speakers: int) -> list[int]:
        import hdbscan
        import numpy as np
        import umap

        if len(embeddings) <= 2:
            return [0] * len(embeddings)

        n_neighbors = max(2, min(16, len(embeddings) - 1))
        umap_embeddings = umap.UMAP(
            n_components=min(32, len(embeddings) - 2),
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=0.05,
            random_state=2023,
            n_jobs=1,
        ).fit_transform(np.array(embeddings))

        labels = hdbscan.HDBSCAN(
            allow_single_cluster=True,
            min_cluster_size=4,
            approx_min_span_tree=False,
            core_dist_n_jobs=1,
        ).fit_predict(umap_embeddings)

        # PAHC (official WeSpeaker) para merge de clusters
        return list(
            WespeakerSpeakerDiarizationAdapter._pahc(
                labels, np.array(embeddings), merge_cutoff=0.3, min_cluster_size=3, absorb_cutoff=0.0
            )
        )

    @staticmethod
    def _pahc(
        labels: list[int],
        embeddings: Any,
        *,
        merge_cutoff: float,
        min_cluster_size: int,
        absorb_cutoff: float,
    ) -> list[int]:
        import heapq
        from collections import defaultdict

        import numpy as np

        class _PAHC:
            def __init__(self) -> None:
                self.labels = list(labels)
                self.embeddings = embeddings
                self.active_clusters: set[int] = set()
                self.label_map: dict[int, list[int]] = defaultdict(list)
                self.cost_map: dict[tuple[int, int], float] = {}
                self.heap: list[tuple[float, tuple[int, int]]] = []
                self.next_index = -1
                self.num_labeled = 0

            @staticmethod
            def _l2norm(x: np.ndarray) -> np.ndarray:
                return x / np.linalg.norm(x, axis=0, keepdims=True)

            def _build_label_map(self) -> None:
                for i, label in enumerate(self.labels):
                    self.label_map[label].append(i)
                self.num_labeled = len(self.label_map)
                if -1 in self.label_map:
                    self.num_labeled -= 1
                    start = self.num_labeled
                    for j, idx in enumerate(self.label_map[-1]):
                        self.label_map[start + j].append(idx)
                    del self.label_map[-1]

            def _compute_cost(self, i_indexes: list[int], j_indexes: list[int]) -> float:
                i_emb = sum(self._l2norm(self.embeddings[i]) for i in i_indexes)
                j_emb = sum(self._l2norm(self.embeddings[j]) for j in j_indexes)
                return float(np.dot(i_emb, j_emb))

            def _build_cost_map(self) -> None:
                n = len(self.label_map)
                self.active_clusters = set(range(n))
                self.next_index = n
                for i in range(n):
                    for j in range(i + 1, n):
                        ii, jj = self.label_map[i], self.label_map[j]
                        if i < self.num_labeled and j < self.num_labeled:
                            self.cost_map[(i, j)] = -np.inf
                            continue
                        self.cost_map[(i, j)] = self._compute_cost(ii, jj)
                        factor = len(ii) * len(jj)
                        if factor and self.cost_map[(i, j)] / factor >= merge_cutoff:
                            heapq.heappush(self.heap, (-self.cost_map[(i, j)] / factor, (i, j)))

            def _merge(self, i: int, j: int) -> None:
                ii, jj = self.label_map[i], self.label_map[j]
                for k in list(self.label_map.keys()):
                    if k in (i, j):
                        continue
                    pair1 = (k, i) if k < i else (i, k)
                    pair2 = (k, j) if k < j else (j, k)
                    cost = self.cost_map.get(pair1, 0.0) + self.cost_map.get(pair2, 0.0)
                    self.cost_map[(k, self.next_index)] = cost
                    factor = (len(ii) + len(jj)) * len(self.label_map[k])
                    if factor and cost / factor >= merge_cutoff:
                        heapq.heappush(self.heap, (-cost / factor, (k, self.next_index)))
                self.label_map[self.next_index] = ii + jj
                self.active_clusters.add(self.next_index)
                for x in (i, j):
                    del self.label_map[x]
                    self.active_clusters.discard(x)
                self.next_index += 1

            def _eliminate(self, i: int) -> None:
                del self.label_map[i]
                self.active_clusters.discard(i)

            def _absorb(self) -> None:
                minor = {k for k, v in self.label_map.items() if len(v) < min_cluster_size}
                major = {k for k, v in self.label_map.items() if len(v) >= min_cluster_size}
                if not major:
                    return
                for i in minor:
                    max_cost = -np.inf
                    closest = -1
                    for j in major:
                        pair = (i, j) if i < j else (j, i)
                        if pair not in self.cost_map:
                            continue
                        factor = len(self.label_map[i]) * len(self.label_map[j])
                        if factor and self.cost_map[pair] / factor > max_cost:
                            max_cost = self.cost_map[pair] / factor
                            closest = j
                    if closest >= 0 and max_cost >= absorb_cutoff:
                        self.label_map[closest].extend(self.label_map[i])
                        self._eliminate(i)

            def _relabel(self) -> list[int]:
                out = [-1] * len(self.labels)
                for label, indexes in self.label_map.items():
                    for idx in indexes:
                        out[idx] = label
                remap: dict[int, int] = {}
                counter = 0
                for label in out:
                    if label not in remap:
                        remap[label] = counter
                        counter += 1
                return [remap[label] for label in out]

            def run(self) -> list[int]:
                self._build_label_map()
                self._build_cost_map()
                while self.heap:
                    _, (i, j) = heapq.heappop(self.heap)
                    if i in self.active_clusters and j in self.active_clusters:
                        self._merge(i, j)
                self._absorb()
                return self._relabel()

        return _PAHC().run()

    @staticmethod
    def _merge_segments(
        subseg2label: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        if not subseg2label:
            return []
        merged: list[tuple[float, float, int]] = []
        begin, end, label = subseg2label[0]
        e = end
        for (b, e, la) in subseg2label[1:]:
            if b <= end and la == label:
                end = e
            elif b > end:
                merged.append((begin, end, label))
                begin, end, label = b, e, la
            elif b <= end and la != label:
                pivot = (b + end) / 2.0
                merged.append((begin, pivot, label))
                begin, end, label = pivot, e, la
        merged.append((begin, e, label))
        return merged

    @staticmethod
    def _normalize_to_speaker_labels(
        merged: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, str]]:
        mapping: dict[int, str] = {}
        next_index = 0
        out: list[tuple[float, float, str]] = []
        for begin_s, end_s, label in merged:
            if label not in mapping:
                mapping[label] = f"SPEAKER_{next_index:02d}"
                next_index += 1
            out.append((begin_s, end_s, mapping[label]))
        return out

    @staticmethod
    def _resolve_overlaps(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        turns = sorted(turns, key=lambda t: (t.start_ms, t.end_ms))
        resolved: list[SpeakerTurn] = []
        for t in turns:
            if not resolved:
                resolved.append(t)
                continue
            last = resolved[-1]
            if t.start_ms < last.end_ms:
                new_start = last.end_ms
                if new_start >= t.end_ms:
                    continue
                t = SpeakerTurn(
                    speaker_label=t.speaker_label,
                    start_ms=new_start,
                    end_ms=t.end_ms,
                    confidence=t.confidence,
                )
            resolved.append(t)
        return resolved


__all__ = [
    "MODEL_FOLDERNAME",
    "WESPEAKER_SUBFOLDER",
    "WESPEAKER_CHECKPOINT_FILENAME",
    "WespeakerDiarizationModelLoader",
    "WespeakerSpeakerDiarizationAdapter",
]
