"""Deterministic Indian registration grammar and optical confusion repair.

Grammar
-------
Three families cover essentially the whole national fleet:

* **BharatSeries (BH)** - ``YY BH #### XX``: ``25BH1234AB``.
* **Standard / HSRP**  - ``SS DD SSS ####``: state code, RTO district code,
  optional 1-3 letter series, 4-digit number. ``MH12AB1234``, ``DL8CAF5031``,
  ``KA05MH9999``.
* **Legacy short forms** - the series group may be absent entirely
  (``MH121234``), still current on older commercial vehicles.

Each is expanded into concrete *templates* - one per achievable string length -
where every position is declared as alphabetic, numeric, or a literal. That
expansion is what makes repair tractable: once the length is known, the class
of every position is known, so an OCR confusion is either legal or it is not.

Repair
------
A recogniser that emits ``MH12A81234`` has almost certainly seen ``B`` and
written ``8``; the plate is illegal as written. Rather than an arbitrary edit
distance, each substitution is priced in log-probability using the per-character
posteriors recovered by CTC forced alignment:

.. math::
   \\text{cost}(i, a \\to b) = \\log P(a \\mid t_i) - \\log P(b \\mid t_i) \\ge 0

A best-first (uniform-cost) search over substitutions returns the *cheapest*
legal string. Because the cost is a genuine log-likelihood ratio, the total
repair cost is directly comparable across plates and is exported downstream so
Stage 3 can discount a heavily-repaired reading in the confidence fusion.

This is strictly a repair of the *character class*, never of the plate itself:
only confusion pairs that are optically plausible are ever considered, and the
repair budget is capped. A plate that cannot be repaired within budget is
emitted as-is with ``is_valid_format = False`` - suppressing it would be worse,
because Re-ID (Stage 3) can still track the vehicle visually.
"""

from __future__ import annotations

import heapq
import logging
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .config import ValidationConfig
from .types import CharPosterior

logger = logging.getLogger(__name__)

# Union Territory and State RTO prefixes (post-2019 reorganisation).
STATE_CODES: FrozenSet[str] = frozenset(
    {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
        "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
        "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
        "UA", "UP", "WB",
    }
)

ALPHA = "A"
DIGIT = "D"

# Fast pre-check patterns. The template machinery is authoritative; these exist
# to short-circuit the common case without allocating.
RE_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
RE_BHARAT = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Optically confusable classes, curated for the fonts actually seen on Indian
# plates (HSRP embossed, Charles Wright derivatives, and non-standard vinyl).
CONFUSION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("0", "O"), ("0", "D"), ("0", "Q"), ("0", "U"),
    ("1", "I"), ("1", "L"), ("1", "J"), ("1", "T"),
    ("2", "Z"), ("4", "A"), ("5", "S"), ("6", "G"),
    ("7", "T"), ("8", "B"), ("9", "G"), ("9", "Q"),
    ("B", "R"), ("C", "G"), ("E", "F"), ("K", "X"),
    ("M", "N"), ("O", "D"), ("O", "Q"), ("U", "V"),
)


def _build_confusion_map() -> Dict[str, Tuple[str, ...]]:
    table: Dict[str, set] = {}
    for a, b in CONFUSION_PAIRS:
        table.setdefault(a, set()).add(b)
        table.setdefault(b, set()).add(a)
    return {k: tuple(sorted(v)) for k, v in table.items()}


CONFUSION_MAP: Dict[str, Tuple[str, ...]] = _build_confusion_map()


@dataclass(frozen=True, slots=True)
class PlateTemplate:
    """A length-resolved positional grammar."""

    template_id: str
    positions: Tuple[str, ...]  # 'A', 'D', or a literal character
    family: str
    state_code_slice: Optional[Tuple[int, int]] = None

    @property
    def length(self) -> int:
        return len(self.positions)

    def accepts_char(self, index: int, char: str) -> bool:
        spec = self.positions[index]
        if spec == ALPHA:
            return char.isalpha()
        if spec == DIGIT:
            return char.isdigit()
        return char == spec

    def matches(self, text: str) -> bool:
        if len(text) != self.length:
            return False
        return all(self.accepts_char(i, c) for i, c in enumerate(text))


def _build_templates() -> Tuple[PlateTemplate, ...]:
    templates: List[PlateTemplate] = []

    # Standard: [A A] [D{1,2}] [A{0,3}] [D{4}]
    for district_digits in (1, 2):
        for series_letters in (0, 1, 2, 3):
            positions = (
                (ALPHA,) * 2
                + (DIGIT,) * district_digits
                + (ALPHA,) * series_letters
                + (DIGIT,) * 4
            )
            templates.append(
                PlateTemplate(
                    template_id=f"IN-STD-{district_digits}{series_letters}",
                    positions=positions,
                    family="standard",
                    state_code_slice=(0, 2),
                )
            )

    # Bharat series: [D D] 'B' 'H' [D{4}] [A{1,2}]
    for suffix_letters in (1, 2):
        positions = (DIGIT, DIGIT, "B", "H") + (DIGIT,) * 4 + (ALPHA,) * suffix_letters
        templates.append(
            PlateTemplate(
                template_id=f"IN-BH-{suffix_letters}",
                positions=positions,
                family="bharat",
            )
        )

    return tuple(templates)


TEMPLATES: Tuple[PlateTemplate, ...] = _build_templates()

_TEMPLATES_BY_LENGTH: Dict[int, Tuple[PlateTemplate, ...]] = {}
for _template in TEMPLATES:
    _TEMPLATES_BY_LENGTH.setdefault(_template.length, ())
    _TEMPLATES_BY_LENGTH[_template.length] += (_template,)


def normalise_text(text: str) -> str:
    """Strip separators and upper-case; plates are stored canonically."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def match_template(text: str) -> Optional[PlateTemplate]:
    """Return the first template accepting ``text``, else ``None``.

    Templates are ordered so that longer series groups are tried before shorter
    ones, which is unambiguous because all templates of a given length differ in
    at least one positional class.
    """
    for template in _TEMPLATES_BY_LENGTH.get(len(text), ()):
        if template.matches(text):
            return template
    return None


def extract_state_code(text: str, template: PlateTemplate) -> Optional[str]:
    if template.state_code_slice is None:
        return None
    start, end = template.state_code_slice
    return text[start:end]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of grammar validation and, where needed, repair."""

    text: str
    raw_text: str
    template_id: str
    is_valid: bool
    was_repaired: bool
    repair_cost: float
    state_code: Optional[str]
    state_code_known: bool


def _posterior_lookup(posterior: CharPosterior) -> Dict[str, float]:
    table = {posterior.char: posterior.log_prob}
    for char, score in zip(posterior.alt_chars, posterior.alt_log_probs):
        table[char] = score
    return table


class PlateValidator:
    """Grammar checker with posterior-priced optical confusion repair."""

    __slots__ = ("_config",)

    def __init__(self, config: ValidationConfig) -> None:
        self._config = config

    def validate(
        self,
        text: str,
        posteriors: Sequence[CharPosterior] = (),
    ) -> ValidationResult:
        """Validate ``text``, repairing optical confusions where affordable.

        Args:
            text: the decoded plate, in any case, with or without separators.
            posteriors: per-character posteriors from CTC forced alignment. When
                empty, repair falls back to a uniform substitution cost of 1.0
                so the search still prefers minimal edits.
        """
        raw = normalise_text(text)
        if not raw:
            return ValidationResult("", raw, "UNKNOWN", False, False, 0.0, None, False)

        direct = match_template(raw)
        if direct is not None:
            state = extract_state_code(raw, direct)
            known = state is None or state in STATE_CODES
            if known or not self._config.enforce_state_code:
                return ValidationResult(raw, raw, direct.template_id, True, False, 0.0, state, known)

        repaired = self._repair(raw, posteriors)
        if repaired is not None:
            candidate, cost, template = repaired
            state = extract_state_code(candidate, template)
            known = state is None or state in STATE_CODES
            return ValidationResult(candidate, raw, template.template_id, True, True, cost, state, known)

        if direct is not None:
            state = extract_state_code(raw, direct)
            return ValidationResult(raw, raw, direct.template_id, True, False, 0.0, state, False)
        return ValidationResult(raw, raw, "UNKNOWN", False, False, 0.0, None, False)

    def _substitution_cost(
        self, index: int, original: str, replacement: str, posteriors: Sequence[CharPosterior]
    ) -> float:
        """Log-likelihood ratio cost of swapping one character."""
        if index >= len(posteriors):
            return 1.0
        table = _posterior_lookup(posteriors[index])
        if replacement not in table:
            # Outside the recorded top-k: charge the gap to the weakest recorded
            # alternative plus a margin, so unseen substitutions are dispreferred
            # but not impossible.
            weakest = min(table.values()) if table else 0.0
            return max(0.0, table.get(original, 0.0) - weakest) + 1.5
        cost = table.get(original, 0.0) - table[replacement]
        return float(max(cost, 0.0))

    def _repair(
        self, text: str, posteriors: Sequence[CharPosterior]
    ) -> Optional[Tuple[str, float, PlateTemplate]]:
        """Uniform-cost search for the cheapest grammar-legal repair."""
        candidates = _TEMPLATES_BY_LENGTH.get(len(text))
        if not candidates:
            return None

        best: Optional[Tuple[str, float, PlateTemplate]] = None
        cfg = self._config

        for template in candidates:
            # Positions that already satisfy the template need no work.
            violations = [i for i, c in enumerate(text) if not template.accepts_char(i, c)]
            if len(violations) > cfg.max_substitutions:
                continue
            if not violations:
                state = extract_state_code(text, template)
                known = state is None or state in STATE_CODES
                penalty = 0.0 if known else cfg.unknown_state_penalty
                if best is None or penalty < best[1]:
                    best = (text, penalty, template)
                continue

            result = self._search_template(text, template, violations, posteriors)
            if result is not None and (best is None or result[1] < best[1]):
                best = (result[0], result[1], template)

        if best is None or best[1] > cfg.max_repair_cost:
            return None
        return best

    def _search_template(
        self,
        text: str,
        template: PlateTemplate,
        violations: Sequence[int],
        posteriors: Sequence[CharPosterior],
    ) -> Optional[Tuple[str, float]]:
        """Best-first search repairing exactly the violating positions."""
        cfg = self._config
        chars = list(text)

        # Enumerate legal replacements per violating position, cheapest first.
        options: List[List[Tuple[float, str]]] = []
        for index in violations:
            original = chars[index]
            spec = template.positions[index]
            allowed: List[Tuple[float, str]] = []
            pool = CONFUSION_MAP.get(original, ())
            for replacement in pool:
                if spec == ALPHA and not replacement.isalpha():
                    continue
                if spec == DIGIT and not replacement.isdigit():
                    continue
                if spec not in (ALPHA, DIGIT) and replacement != spec:
                    continue
                allowed.append(
                    (self._substitution_cost(index, original, replacement, posteriors), replacement)
                )
            if spec not in (ALPHA, DIGIT):
                # Literal position: the only legal value is the literal itself.
                allowed = [
                    (self._substitution_cost(index, original, spec, posteriors), spec)
                ]
            if not allowed:
                return None
            allowed.sort()
            options.append(allowed)

        # Uniform-cost expansion over the cartesian product, cheapest-first.
        start = tuple(0 for _ in options)
        initial_cost = sum(opt[0][0] for opt in options)
        heap: List[Tuple[float, Tuple[int, ...]]] = [(initial_cost, start)]
        seen = {start}
        expansions = 0

        while heap and expansions < cfg.max_search_expansions:
            cost, state_indices = heapq.heappop(heap)
            expansions += 1
            if cost > cfg.max_repair_cost:
                return None

            candidate = chars.copy()
            for slot, (position, choice) in enumerate(zip(violations, state_indices)):
                candidate[position] = options[slot][choice][1]
            candidate_text = "".join(candidate)

            if template.matches(candidate_text):
                total = cost
                state = extract_state_code(candidate_text, template)
                if state is not None and state not in STATE_CODES:
                    if cfg.enforce_state_code:
                        total += cfg.unknown_state_penalty
                if total <= cfg.max_repair_cost:
                    return candidate_text, float(total)

            for slot in range(len(state_indices)):
                nxt = state_indices[slot] + 1
                if nxt >= len(options[slot]):
                    continue
                successor = state_indices[:slot] + (nxt,) + state_indices[slot + 1 :]
                if successor in seen:
                    continue
                seen.add(successor)
                delta = options[slot][nxt][0] - options[slot][state_indices[slot]][0]
                heapq.heappush(heap, (cost + delta, successor))

        return None
