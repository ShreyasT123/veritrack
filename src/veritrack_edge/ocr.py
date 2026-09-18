"""CTC recognition, multi-frame logit fusion, and decoding.

Multi-frame tracklet fusion
---------------------------
Every observation of a tracklet is warped onto the *same* canonical strip, so
CTC time-step :math:`t` corresponds to the same horizontal band of the physical
plate in every frame. That is what makes logit-level fusion valid: the
sequences are aligned by construction, not by assumption.

Frame :math:`f` contributes posterior :math:`P_f(t, c)` with weight
:math:`w_f`, and the fused posterior is the weighted mixture

.. math::
   P^{*}(t, c) = \\sum_{f} w_f\\, P_f(t, c), \\qquad \\sum_f w_f = 1

computed in log space as :math:`\\log P^{*} = \\operatorname{logsumexp}_f
(\\log w_f + \\log P_f)`. Mixture rather than product: a product (log-opinion
pool) lets a single confidently-wrong frame - a specular highlight burning out
one character - veto the other eleven, whereas a mixture degrades gracefully.

The weight combines three independent quality signals:

.. math::
   w_f \\propto q_{\\text{focus}}^{\\alpha}\\;
               s_{\\text{det}}^{\\beta}\\;
               e^{-\\lambda \\bar{H}_f}, \\qquad
   \\bar{H}_f = \\frac{1}{T\\ln C}\\sum_{t}\\sum_{c} -P_f(t,c)\\ln P_f(t,c)

:math:`\\bar{H}_f` is the mean per-step entropy normalised to ``[0, 1]``. A
frame the recogniser is uncertain about is exponentially discounted. The same
entropy is carried forward: Stage 3's fusion weight
:math:`w_{\\text{text}}` is driven by the entropy of the *fused* sequence.

Decoding
--------
CTC prefix beam search over the fused posterior, then Viterbi forced alignment
of the winning string to recover per-character posteriors, which the grammar
repairer (:mod:`veritrack_edge.validation`) needs in order to price a
substitution in log-probability rather than by an arbitrary edit cost.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import BackendConfig, OcrConfig
from .errors import ModelContractError, RecognitionError
from .gating import normalized_focus
from .runtime import InferenceBackend, load_backend
from .types import CharPosterior, PlateHypothesis, PlateObservation

logger = logging.getLogger(__name__)

_NEG_INF = -1.0e30


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return (shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))).astype(np.float32)


def sequence_entropy(log_probs: np.ndarray) -> float:
    """Mean per-step Shannon entropy, normalised to ``[0, 1]`` by ``ln C``."""
    probs = np.exp(log_probs.astype(np.float64))
    per_step = -np.sum(probs * np.where(probs > 0.0, np.log(probs), 0.0), axis=1)
    num_classes = log_probs.shape[1]
    if num_classes < 2:
        return 0.0
    return float(np.clip(per_step.mean() / math.log(num_classes), 0.0, 1.0))


def fuse_log_probs(
    log_prob_stack: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """Weighted probability-space mixture of aligned per-frame posteriors.

    Args:
        log_prob_stack: per-frame ``(T, C)`` log-softmax arrays, all same shape.
        weights: non-negative, same length; renormalised internally.

    Raises:
        RecognitionError: empty input, shape mismatch, or zero total weight.
    """
    if not log_prob_stack:
        raise RecognitionError("Cannot fuse an empty observation set")
    if len(log_prob_stack) != len(weights):
        raise RecognitionError("Weight count does not match observation count")

    reference = log_prob_stack[0].shape
    for arr in log_prob_stack:
        if arr.shape != reference:
            raise RecognitionError(
                f"Observation shape mismatch: {arr.shape} vs {reference}; "
                "tracklet fusion requires a fixed canonical strip"
            )

    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0.0):
        raise RecognitionError("Fusion weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        w = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    else:
        w = w / total

    stack = np.stack([a.astype(np.float64) for a in log_prob_stack], axis=0)  # (F, T, C)
    log_w = np.log(np.maximum(w, 1e-12)).reshape(-1, 1, 1)
    fused = _logsumexp(stack + log_w, axis=0)
    # Renormalise: floating-point drift can leave rows off the simplex.
    fused = fused - _logsumexp(fused, axis=1, keepdims=True)
    return fused.astype(np.float32)


def _logsumexp(arr: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    peak = np.max(arr, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak + np.log(np.sum(np.exp(arr - peak), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _logaddexp(a: float, b: float) -> float:
    if a <= _NEG_INF:
        return b
    if b <= _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def observation_weight(
    observation: PlateObservation,
    config: OcrConfig,
    focus_reference: float,
) -> float:
    """Quality weight for one observation in the tracklet mixture."""
    focus_q = normalized_focus(observation.focus_score, focus_reference)
    det_q = float(np.clip(observation.detection_score, 0.0, 1.0))
    entropy = sequence_entropy(observation.log_probs)
    weight = (
        max(focus_q, 1e-4) ** config.focus_exponent
        * max(det_q, 1e-4) ** config.detection_conf_exponent
        * math.exp(-config.entropy_lambda * entropy)
    )
    return float(max(weight, 1e-9))


# ---------------------------------------------------------------------------
# CTC decoding
# ---------------------------------------------------------------------------


def ctc_prefix_beam_search(
    log_probs: np.ndarray,
    charset: str,
    blank_index: int = 0,
    beam_width: int = 10,
    topk_per_step: int = 6,
) -> List[PlateHypothesis]:
    """Decode with CTC prefix beam search.

    Maintains, for each prefix, the split probabilities :math:`p_b` (paths
    ending in blank) and :math:`p_{nb}` (ending in the last emitted label).
    The split is what allows a repeated character (``MH11`` -> ``11``) to be
    emitted correctly: a repeat may only extend a prefix through :math:`p_b`.

    Args:
        log_probs: ``(T, C)`` log-softmax.
        charset: ``C - 1`` characters; index ``i`` of the charset maps to class
            ``i + 1`` when ``blank_index == 0``.
        beam_width: surviving prefixes per step.
        topk_per_step: classes considered per step (pruning).

    Returns:
        Hypotheses sorted by descending log-probability.

    Raises:
        ModelContractError: the class dimension disagrees with the charset.
    """
    if log_probs.ndim != 2:
        raise ModelContractError(f"CTC input must be (T, C), got {log_probs.shape}")
    num_steps, num_classes = log_probs.shape
    if num_classes != len(charset) + 1:
        raise ModelContractError(
            f"Charset of {len(charset)} chars implies {len(charset) + 1} classes, "
            f"model emits {num_classes}"
        )
    if num_steps == 0:
        return []

    index_to_char = _build_index_map(charset, blank_index)
    k = int(min(max(1, topk_per_step), num_classes))
    lp = log_probs.astype(np.float64)

    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, _NEG_INF)}
    for t in range(num_steps):
        candidates = np.argpartition(-lp[t], k - 1)[:k] if k < num_classes else np.arange(num_classes)
        nxt: Dict[Tuple[int, ...], Tuple[float, float]] = defaultdict(lambda: (_NEG_INF, _NEG_INF))
        for prefix, (p_blank, p_non_blank) in beams.items():
            p_total = _logaddexp(p_blank, p_non_blank)
            last = prefix[-1] if prefix else None
            for cls in candidates:
                cls = int(cls)
                p = float(lp[t, cls])
                if cls == blank_index:
                    b, nb = nxt[prefix]
                    nxt[prefix] = (_logaddexp(b, p_total + p), nb)
                    continue
                if cls == last:
                    # Same label again: collapses unless separated by a blank.
                    b, nb = nxt[prefix]
                    nxt[prefix] = (b, _logaddexp(nb, p_non_blank + p))
                    extended = prefix + (cls,)
                    eb, enb = nxt[extended]
                    nxt[extended] = (eb, _logaddexp(enb, p_blank + p))
                else:
                    extended = prefix + (cls,)
                    eb, enb = nxt[extended]
                    nxt[extended] = (eb, _logaddexp(enb, p_total + p))

        beams = dict(
            sorted(
                nxt.items(),
                key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    results: List[PlateHypothesis] = []
    for prefix, (p_blank, p_non_blank) in beams.items():
        score = _logaddexp(p_blank, p_non_blank)
        text = "".join(index_to_char[i] for i in prefix)
        if text:
            results.append(PlateHypothesis(text=text, log_prob=float(score)))
    results.sort(key=lambda h: h.log_prob, reverse=True)
    return results


def ctc_greedy_decode(log_probs: np.ndarray, charset: str, blank_index: int = 0) -> str:
    """Best-path decode; used as a cheap sanity path and in unit tests."""
    index_to_char = _build_index_map(charset, blank_index)
    path = np.argmax(log_probs, axis=1)
    out: List[str] = []
    previous = -1
    for cls in path:
        cls = int(cls)
        if cls != previous and cls != blank_index:
            out.append(index_to_char[cls])
        previous = cls
    return "".join(out)


def _build_index_map(charset: str, blank_index: int) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    cursor = 0
    for class_index in range(len(charset) + 1):
        if class_index == blank_index:
            continue
        mapping[class_index] = charset[cursor]
        cursor += 1
    return mapping


def _char_to_class(charset: str, blank_index: int) -> Dict[str, int]:
    return {char: index for index, char in _build_index_map(charset, blank_index).items()}


def ctc_forced_alignment(
    log_probs: np.ndarray,
    text: str,
    charset: str,
    blank_index: int = 0,
    top_alternatives: int = 4,
) -> List[CharPosterior]:
    """Viterbi-align ``text`` to ``log_probs`` and extract per-character posteriors.

    Runs the standard CTC dynamic program over the extended label sequence
    :math:`\\ell' = [\\varnothing, \\ell_1, \\varnothing, \\ell_2, \\dots,
    \\varnothing]` with the max-product (Viterbi) semiring, then reports, for
    each character, the posterior distribution at the single time step where
    that character's probability peaked along the alignment.

    Raises:
        RecognitionError: ``text`` cannot be aligned to ``T`` steps.
    """
    char_to_class = _char_to_class(charset, blank_index)
    try:
        labels = [char_to_class[c] for c in text]
    except KeyError as exc:
        raise RecognitionError(f"Character {exc} is outside the model charset") from exc

    num_steps = log_probs.shape[0]
    extended: List[int] = [blank_index]
    for label in labels:
        extended.extend([label, blank_index])
    num_states = len(extended)
    if num_steps < len(labels):
        raise RecognitionError(
            f"Cannot align {len(labels)} characters to {num_steps} CTC steps"
        )

    lp = log_probs.astype(np.float64)
    dp = np.full((num_steps, num_states), _NEG_INF, dtype=np.float64)
    back = np.zeros((num_steps, num_states), dtype=np.int32)

    dp[0, 0] = lp[0, extended[0]]
    if num_states > 1:
        dp[0, 1] = lp[0, extended[1]]

    for t in range(1, num_steps):
        for s in range(num_states):
            best_prev, best_score = s, dp[t - 1, s]
            if s > 0 and dp[t - 1, s - 1] > best_score:
                best_prev, best_score = s - 1, dp[t - 1, s - 1]
            # A skip over a blank is legal only between distinct labels.
            if (
                s > 1
                and extended[s] != blank_index
                and extended[s] != extended[s - 2]
                and dp[t - 1, s - 2] > best_score
            ):
                best_prev, best_score = s - 2, dp[t - 1, s - 2]
            if best_score <= _NEG_INF:
                continue
            dp[t, s] = best_score + lp[t, extended[s]]
            back[t, s] = best_prev

    terminal = [num_states - 1] + ([num_states - 2] if num_states >= 2 else [])
    end_state = max(terminal, key=lambda s: dp[num_steps - 1, s])
    if dp[num_steps - 1, end_state] <= _NEG_INF:
        raise RecognitionError(f"No valid CTC alignment exists for '{text}'")

    path = np.zeros(num_steps, dtype=np.int32)
    state = end_state
    for t in range(num_steps - 1, -1, -1):
        path[t] = state
        state = int(back[t, state])

    # Peak time step per emitted character (odd indices of the extended seq).
    peaks: Dict[int, Tuple[int, float]] = {}
    for t, state_index in enumerate(path):
        state_index = int(state_index)
        if state_index % 2 == 0:
            continue
        char_position = state_index // 2
        score = float(lp[t, extended[state_index]])
        current = peaks.get(char_position)
        if current is None or score > current[1]:
            peaks[char_position] = (t, score)

    index_to_char = _build_index_map(charset, blank_index)
    posteriors: List[CharPosterior] = []
    for position, char in enumerate(text):
        peak = peaks.get(position)
        if peak is None:
            # Character consumed entirely by blank collapse; fall back to the
            # globally best step for its class.
            cls = char_to_class[char]
            step = int(np.argmax(lp[:, cls]))
            peak = (step, float(lp[step, cls]))
        step, score = peak
        row = lp[step]
        order = np.argsort(-row)
        alt_chars: List[str] = []
        alt_scores: List[float] = []
        for cls in order:
            cls = int(cls)
            if cls == blank_index or index_to_char[cls] == char:
                continue
            alt_chars.append(index_to_char[cls])
            alt_scores.append(float(row[cls]))
            if len(alt_chars) >= top_alternatives:
                break
        posteriors.append(
            CharPosterior(
                char=char,
                log_prob=score,
                alt_chars=tuple(alt_chars),
                alt_log_probs=tuple(alt_scores),
            )
        )
    return posteriors


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusedRecognition:
    """Decoded output for a whole tracklet."""

    hypotheses: Tuple[PlateHypothesis, ...]
    fused_log_probs: np.ndarray
    posteriors: Tuple[CharPosterior, ...]
    entropy: float
    observation_count: int

    @property
    def best(self) -> PlateHypothesis:
        if not self.hypotheses:
            raise RecognitionError("No hypotheses were decoded")
        return self.hypotheses[0]


class CtcRecognizer:
    """SVTR / PP-OCRv6-Tiny CTC recogniser over the canonical strip."""

    __slots__ = ("_config", "_backend", "_focus_reference")

    def __init__(
        self,
        config: OcrConfig,
        backend_config: BackendConfig,
        focus_reference: float,
        backend: Optional[InferenceBackend] = None,
    ) -> None:
        self._config = config
        self._focus_reference = focus_reference
        self._backend = backend if backend is not None else load_backend(config.model_path, backend_config)

    @property
    def backend(self) -> InferenceBackend:
        return self._backend

    @property
    def config(self) -> OcrConfig:
        return self._config

    def preprocess(self, strip: np.ndarray) -> np.ndarray:
        """Canonical strip -> ``(1, 3, H, W)`` float32 tensor."""
        width, height = self._config.input_size
        if strip.ndim == 2:
            strip = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        if (strip.shape[1], strip.shape[0]) != (width, height):
            interp = cv2.INTER_AREA if strip.shape[1] > width else cv2.INTER_CUBIC
            strip = cv2.resize(strip, (width, height), interpolation=interp)
        rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._config.normalize_mean) / self._config.normalize_std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])

    def infer_log_probs(self, strip: np.ndarray) -> np.ndarray:
        """Return ``(T, C)`` log-softmax for one strip.

        Accepts either raw logits or an already-softmaxed head; the two are
        distinguished by checking whether rows sum to one.
        """
        raw = self._backend.run(self.preprocess(strip))[0]
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ModelContractError(f"Recogniser must emit (T, C), got {arr.shape}")
        expected = len(self._config.charset) + 1
        if arr.shape[1] != expected:
            if arr.shape[0] == expected:  # (C, T) export
                arr = arr.transpose(1, 0)
            else:
                raise ModelContractError(
                    f"Recogniser emits {arr.shape[1]} classes, charset implies {expected}"
                )
        row_sums = arr.sum(axis=1)
        if np.all(arr >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-2):
            return np.log(np.clip(arr, 1e-12, None)).astype(np.float32)
        return log_softmax(arr, axis=1)

    def fuse_and_decode(self, observations: Sequence[PlateObservation]) -> FusedRecognition:
        """Fuse a tracklet's observations and decode the result.

        Raises:
            RecognitionError: no observations, or decoding produced nothing.
        """
        if not observations:
            raise RecognitionError("Tracklet has no plate observations")

        selected = sorted(
            observations,
            key=lambda o: observation_weight(o, self._config, self._focus_reference),
            reverse=True,
        )[: self._config.max_observations]

        weights = [observation_weight(o, self._config, self._focus_reference) for o in selected]
        fused = fuse_log_probs([o.log_probs for o in selected], weights)

        hypotheses = ctc_prefix_beam_search(
            fused,
            charset=self._config.charset,
            blank_index=self._config.blank_index,
            beam_width=self._config.beam_width,
            topk_per_step=self._config.topk_per_step,
        )
        if not hypotheses:
            raise RecognitionError("CTC beam search returned no hypotheses")

        posteriors = ctc_forced_alignment(
            fused,
            hypotheses[0].text,
            charset=self._config.charset,
            blank_index=self._config.blank_index,
        )
        return FusedRecognition(
            hypotheses=tuple(hypotheses),
            fused_log_probs=fused,
            posteriors=tuple(posteriors),
            entropy=sequence_entropy(fused),
            observation_count=len(selected),
        )
