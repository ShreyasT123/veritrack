"""Dynamic multi-modal confidence fusion for Stage 3.

Two sightings are scored against each other on three independent channels and
combined:

.. math::  S(i,j) = w_{\\text{text}} S_{\\text{text}} + w_{\\text{vis}} S_{\\text{vis}} + w_{\\text{kin}} S_{\\text{kin}}

The point of the design is that the **weights are not constants**. When the OCR
read is strong, the plate string is the most discriminative evidence available
and should dominate. When it is weak -- a damaged plate, heavy glare, a 45
degree view -- insisting on the text would either link the wrong vehicles or
link none. The gate

.. math::  g = \\frac{1}{1 + e^{-k(C_{\\text{seq}} - \\tau)}}

slides the mass onto the visual embedding exactly when the text stops being
trustworthy.

The schedule has a tidy closed form. Substituting the coefficients:

.. math::  w_{\\text{kin}} = 1 - w_{\\text{text}} - w_{\\text{vis}} = 0.35 - 0.20\\,g

so the three weights always sum to one and :math:`w_{\\text{kin}}` stays in
[0.15, 0.35] -- it never goes negative, at any gate value. That is worth
checking rather than assuming, and ``TrajectoryConfig`` asserts it at both
extremes on construction.

Kinematics is never allowed to dominate, and that is deliberate. Travel speed
is *corroborating* evidence: it can veto an impossible link, but it cannot by
itself assert that two vehicles are the same one, because thousands of vehicles
traverse a corridor at a plausible speed every hour.
"""

from __future__ import annotations

import math
from typing import Dict, Final, Mapping, Optional, Sequence, Tuple

from .config import FusionConfig, TrajectoryConfig
from .types import CameraSighting, MatchScore

__all__ = [
    "CONFUSION_PAIRS",
    "CONFUSION_MAP",
    "damerau_levenshtein",
    "text_similarity",
    "cosine_similarity",
    "visual_similarity",
    "kinematic_similarity",
    "ConfidenceFusion",
]

#: Optical confusion pairs shared with ``veritrack_edge.validation``. These are
#: the substitutions an ANPR recogniser actually makes; a generic edit distance
#: treats "8 -> B" as just as surprising as "8 -> Q", which it is not.
CONFUSION_PAIRS: Final[Tuple[Tuple[str, str], ...]] = (
    ("0", "O"), ("0", "D"), ("0", "Q"),
    ("1", "I"), ("1", "L"), ("1", "T"),
    ("2", "Z"), ("5", "S"), ("6", "G"),
    ("8", "B"), ("4", "A"), ("7", "T"),
    ("9", "G"), ("U", "V"), ("M", "N"),
)


def _build_confusion_map() -> Dict[str, frozenset[str]]:
    mapping: Dict[str, set[str]] = {}
    for left, right in CONFUSION_PAIRS:
        mapping.setdefault(left, set()).add(right)
        mapping.setdefault(right, set()).add(left)
    return {key: frozenset(value) for key, value in mapping.items()}


CONFUSION_MAP: Final[Dict[str, frozenset[str]]] = _build_confusion_map()


def _substitution_cost(left: str, right: str, confusion_cost: float) -> float:
    """Cost of replacing ``left`` with ``right``."""
    if left == right:
        return 0.0
    if right in CONFUSION_MAP.get(left, frozenset()):
        return confusion_cost
    return 1.0


def damerau_levenshtein(
    left: str,
    right: str,
    *,
    confusion_cost: float = 0.30,
    transposition_cost: float = 1.0,
) -> float:
    """Weighted Damerau-Levenshtein distance with optical confusion discounts.

    This is the *restricted* (optimal string alignment) variant: it allows
    adjacent transpositions but does not permit a substring to be edited twice.
    For registration plates -- short, fixed-format strings where a genuine
    transposition is a single reversed character pair -- the restricted and
    unrestricted forms agree, and the restricted one is O(mn) with a clean
    weighted formulation.

    Costs are floats rather than integers precisely so that a confusable
    substitution can cost 0.30 while an arbitrary one costs 1.0. Stage 1
    already priced these confusions in nats and may have repaired them;
    charging full price again here would discard correct matches.
    """
    if left == right:
        return 0.0
    m, n = len(left), len(right)
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)

    previous_previous: list[float] = []
    previous: list[float] = [float(index) for index in range(n + 1)]

    for i in range(1, m + 1):
        current: list[float] = [float(i)] + [0.0] * n
        for j in range(1, n + 1):
            cost = _substitution_cost(left[i - 1], right[j - 1], confusion_cost)
            deletion = previous[j] + 1.0
            insertion = current[j - 1] + 1.0
            substitution = previous[j - 1] + cost
            best = min(deletion, insertion, substitution)
            if (
                i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                best = min(best, previous_previous[j - 2] + transposition_cost)
            current[j] = best
        previous_previous, previous = previous, current
    return previous[n]


def text_similarity(
    left: str,
    right: str,
    *,
    confusion_cost: float = 0.30,
    transposition_cost: float = 1.0,
) -> float:
    """Normalised text similarity in [0, 1]. Exactly 1.0 iff the strings match.

    Normalising by the longer string means the score is comparable across
    plate formats: a one-character error on a 10-character plate costs the same
    fraction whether the plate is standard or a short legacy format.
    """
    normalised_left = left.strip().upper()
    normalised_right = right.strip().upper()
    if not normalised_left and not normalised_right:
        return 0.0
    if normalised_left == normalised_right:
        return 1.0
    longest = max(len(normalised_left), len(normalised_right))
    if longest == 0:
        return 0.0
    distance = damerau_levenshtein(
        normalised_left,
        normalised_right,
        confusion_cost=confusion_cost,
        transposition_cost=transposition_cost,
    )
    return max(0.0, 1.0 - distance / longest)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors, in [-1, 1].

    Stage 1 emits L2-normalised embeddings and Stage 2 re-normalises after
    int8 dequantisation, so in practice this is a dot product. The norms are
    computed anyway: a silently unnormalised vector would inflate every
    similarity it touched, and that failure would be invisible.
    """
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError(
            f"embedding dimension mismatch: {len(left)} vs {len(right)}"
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def visual_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Re-ID similarity clamped to [0, 1] for use as a fusion component.

    A negative cosine means the two appearance vectors point away from each
    other, which is simply "not the same vehicle". Clamping at zero rather
    than rescaling from [-1, 1] keeps the component on the same scale as the
    other two, so the weights mean what they say.
    """
    return max(0.0, cosine_similarity(left, right))


def kinematic_similarity(
    network_distance_m: float,
    delta_t_s: float,
    free_flow_kmh: float,
    *,
    sigma_kmh: float = 18.0,
    max_velocity_kmh: float = 140.0,
    slow_sigma_multiplier: float = 2.5,
    symmetric: bool = False,
) -> Tuple[float, float, bool]:
    """Gaussian feasibility of a transit. Returns (score, speed_kmh, feasible).

    The hard veto comes first: above ``max_velocity_kmh`` the score is exactly
    0.0, because no single vehicle covered that network distance in that time.
    That is what makes the kinematic channel able to *refute* a link no matter
    how good the plate and appearance evidence look -- which is precisely the
    cloned-plate signature.

    Below the veto the score is Gaussian around the corridor free-flow
    baseline, widened on the slow side by ``slow_sigma_multiplier``. See
    ``FusionConfig.kinematic_slow_sigma_multiplier`` for why the symmetric form
    is the wrong physics; pass ``symmetric=True`` to use it anyway.
    """
    if delta_t_s <= 0.0:
        # Zero or negative elapsed time between distinct cameras is either a
        # clock fault or a duplicate. Neither is evidence of a journey.
        return 0.0, 0.0, False
    if not math.isfinite(network_distance_m):
        # Topologically unreachable: no speed is defined, and the link is
        # refuted on Condition C rather than on kinematics.
        return 0.0, math.inf, False

    speed_kmh = (network_distance_m / delta_t_s) * 3.6
    if speed_kmh > max_velocity_kmh:
        return 0.0, speed_kmh, False

    baseline = max(free_flow_kmh, 1e-6)
    effective_sigma = max(sigma_kmh, 1e-6)
    if not symmetric and speed_kmh < baseline:
        effective_sigma *= max(slow_sigma_multiplier, 1e-6)

    deviation = speed_kmh - baseline
    score = math.exp(-(deviation * deviation) / (2.0 * effective_sigma * effective_sigma))
    return min(1.0, max(0.0, score)), speed_kmh, True


class ConfidenceFusion:
    """Scores sighting pairs with dynamically shifting modal weights."""

    __slots__ = ("_config", "_fusion")

    def __init__(self, config: Optional[TrajectoryConfig] = None) -> None:
        self._config = config if config is not None else TrajectoryConfig()
        self._fusion: FusionConfig = self._config.fusion

    @property
    def config(self) -> TrajectoryConfig:
        return self._config

    def gate(self, sequence_confidence: float) -> float:
        """g = sigmoid(k (C_seq - tau)), numerically stable at both tails."""
        return self._config.gate(sequence_confidence)

    def weights(self, sequence_confidence: float) -> Tuple[float, float, float]:
        """(w_text, w_vis, w_kin) for a given OCR sequence confidence."""
        return self._config.weights_for_confidence(sequence_confidence)

    def _pair_confidence(self, left: CameraSighting, right: CameraSighting) -> float:
        """The confidence that drives the gate for a pair.

        The *minimum* of the two, not the mean. The text channel is only as
        trustworthy as the weaker of the two reads being compared: pairing a
        0.98 read with a 0.30 read and gating on 0.64 would hand most of the
        weight to a string that one side barely resolved.
        """
        return min(left.plate_sequence_confidence, right.plate_sequence_confidence)

    def score(
        self,
        left: CameraSighting,
        right: CameraSighting,
        *,
        network_distance_m: float,
        free_flow_kmh: Optional[float] = None,
    ) -> MatchScore:
        """Fuse the three channels into a single match score for a sighting pair."""
        delta_t_s = left.seconds_to(right)
        confidence = self._pair_confidence(left, right)
        gate_value = self.gate(confidence)
        w_text, w_vis, w_kin = self._config.weights_for_gate(gate_value)

        left_text = left.plate_text or left.plate_pseudonym
        right_text = right.plate_text or right.plate_pseudonym
        s_text = text_similarity(
            left_text,
            right_text,
            confusion_cost=self._fusion.confusion_substitution_cost,
            transposition_cost=self._fusion.transposition_cost,
        )

        s_vis = visual_similarity(left.reid_embedding, right.reid_embedding)

        baseline = (
            free_flow_kmh if free_flow_kmh is not None else self._fusion.default_free_flow_kmh
        )
        s_kin, speed_kmh, feasible = kinematic_similarity(
            network_distance_m,
            delta_t_s,
            baseline,
            sigma_kmh=self._fusion.kinematic_sigma_kmh,
            max_velocity_kmh=self._fusion.max_velocity_kmh,
            slow_sigma_multiplier=self._fusion.kinematic_slow_sigma_multiplier,
            symmetric=self._fusion.symmetric_kinematic,
        )

        total = w_text * s_text + w_vis * s_vis + w_kin * s_kin
        return MatchScore(
            total=min(1.0, max(0.0, total)),
            s_text=s_text,
            s_vis=s_vis,
            s_kin=s_kin,
            w_text=w_text,
            w_vis=w_vis,
            w_kin=w_kin,
            gate=gate_value,
            network_distance_m=network_distance_m,
            delta_t_s=delta_t_s,
            network_speed_kmh=speed_kmh if math.isfinite(speed_kmh) else 0.0,
            kinematically_feasible=feasible,
        )

    def is_match(
        self,
        left: CameraSighting,
        right: CameraSighting,
        *,
        network_distance_m: float,
        free_flow_kmh: Optional[float] = None,
    ) -> Tuple[bool, MatchScore]:
        """Score a pair and apply the acceptance threshold."""
        score = self.score(
            left, right, network_distance_m=network_distance_m, free_flow_kmh=free_flow_kmh
        )
        return score.total >= self._config.min_match_score, score

    def explain(self, score: MatchScore) -> str:
        """A one-line human-readable justification, for analyst review."""
        contributions = {
            "text": score.w_text * score.s_text,
            "visual": score.w_vis * score.s_vis,
            "kinematic": score.w_kin * score.s_kin,
        }
        dominant = max(contributions, key=lambda key: contributions[key])
        return (
            f"S={score.total:.3f} driven by {dominant} "
            f"(text {score.s_text:.2f}x{score.w_text:.2f}, "
            f"visual {score.s_vis:.2f}x{score.w_vis:.2f}, "
            f"kinematic {score.s_kin:.2f}x{score.w_kin:.2f}); "
            f"{score.network_speed_kmh:.0f} km/h over "
            f"{score.network_distance_m:.0f} m in {score.delta_t_s:.0f} s"
        )
