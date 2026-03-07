#!/usr/bin/env python3
"""
evaluation.py — Outcome evaluation and retain / discard decisions for src.cbr

This module implements the "Revise" and "Retain" steps of a simple Case-Based
Reasoning (CBR) loop. It is intentionally lightweight and avoids heavy
dependencies or side-effects at import time.

Public functions
----------------
- record_outcome(case_id, outcome, library)
    Attach an outcome dict to an existing case in the CaseLibrary.

- compute_quality_score(outcome) -> float
    Produce a normalized quality score (0.0 - 1.0) from an outcome dict.
    This is a small heuristic useful for gating retention.

- should_retain(case, *, min_quality=0.5, quality_key="quality_score") -> bool
    Decide whether to retain a candidate solved case based on its outcome.

- retain_case(case_id, problem, solution, outcome, library, *, min_quality=0.5) -> bool
    Evaluate outcome and conditionally store the solved case in the CaseLibrary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .case_library import CaseLibrary

logger = logging.getLogger(__name__)


def record_outcome(case_id: str, outcome: Dict[str, Any], library: CaseLibrary) -> None:
    """
    Attach or update the outcome for an existing case in the case library.

    Parameters
    ----------
    case_id:
        Identifier of the case to update.
    outcome:
        Free-form outcome metadata. Expected to be a dict; may contain
        numeric quality indicators (e.g., "quality_score"), textual notes,
        reviewer identifiers, and timestamps.
    library:
        Instance of CaseLibrary to update.

    Raises
    ------
    KeyError
        If the specified case_id does not exist in the library.
    """
    existing = library.get(case_id)
    if existing is None:
        raise KeyError(f"Case '{case_id}' not found in the library.")
    # Merge or replace existing outcome: here we replace with the provided outcome.
    library.add(
        case_id=case_id,
        problem=existing["problem"],
        solution=existing["solution"],
        outcome=outcome,
    )
    logger.debug("Recorded outcome for case '%s': %s", case_id, outcome)


def compute_quality_score(outcome: Dict[str, Any]) -> float:
    """
    Compute a normalized quality score in [0.0, 1.0] from an outcome dict.

    This function is intentionally simple and conservative. It:
      - Uses an explicit numeric 'quality_score' key if present (and numeric),
      - Otherwise, tries to infer a score from common keys (e.g., 'improved', 'rating'),
      - Falls back to 0.5 when no information is present (neutral).

    The heuristic is deliberately modest; teams should replace it with a
    domain-specific metric when available.

    Parameters
    ----------
    outcome:
        The outcome dictionary recorded after applying/adapting a solution.

    Returns
    -------
    float
        Quality score between 0.0 and 1.0.
    """
    if not isinstance(outcome, dict):
        return 0.5

    # 1) Explicit numeric key
    if "quality_score" in outcome:
        try:
            val = float(outcome["quality_score"])
            return max(0.0, min(1.0, val))
        except Exception:
            pass

    # 2) Common inference: boolean improved / success keys
    for k in ("improved", "success", "resolved"):
        if k in outcome:
            v = outcome[k]
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            # parse strings like "yes"/"no", "true"/"false"
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in ("yes", "y", "true", "t", "1"):
                    return 1.0
                if vl in ("no", "n", "false", "f", "0"):
                    return 0.0

    # 3) Numeric rating (e.g., 1-5 stars) try to normalize
    for k in ("rating", "score", "review_score"):
        if k in outcome:
            try:
                val = float(outcome[k])
                # if a typical 1-5 scale is used, try to normalize heuristically
                if 0.0 <= val <= 1.0:
                    return max(0.0, min(1.0, val))
                if 1.0 <= val <= 5.0:
                    return max(0.0, min(1.0, val / 5.0))
                # otherwise clamp into [0,1]
                return max(0.0, min(1.0, val))
            except Exception:
                continue

    # 4) No usable signals: neutral default
    return 0.5


def should_retain(
    case: Dict[str, Any],
    *,
    min_quality: float = 0.5,
    quality_key: str = "quality_score",
) -> bool:
    """
    Decide whether a solved case should be retained in the case library.

    Parameters
    ----------
    case:
        A full case dict. Expected keys: 'problem', 'solution', optionally 'outcome'.
    min_quality:
        Minimum acceptable quality score (0.0 - 1.0). Cases below this threshold
        are rejected.
    quality_key:
        The key inside case['outcome'] to prefer when present.

    Returns
    -------
    bool
        True if the case should be retained; False otherwise.

    Behavior
    --------
    - If no outcome is present, returns True so new solved cases can be reviewed later.
    - If an explicit numeric quality_key exists, it's used directly.
    - Otherwise compute_quality_score(...) is used to infer a score.
    """
    outcome = (case or {}).get("outcome")
    if not outcome:
        # No evaluation yet — retain by default for later human review.
        logger.debug(
            "No outcome present for case '%s' — defaulting to retain",
            case.get("case_id"),
        )
        return True

    # Prefer explicit key if present
    if isinstance(outcome, dict) and quality_key in outcome:
        try:
            score = float(outcome[quality_key])
        except Exception:
            score = compute_quality_score(outcome)
    else:
        score = compute_quality_score(outcome)

    logger.debug(
        "Evaluated quality score for case '%s': %s (min required: %s)",
        case.get("case_id"),
        score,
        min_quality,
    )
    return float(score) >= float(min_quality)


def retain_case(
    case_id: str,
    problem: Dict[str, Any],
    solution: Dict[str, Any],
    outcome: Dict[str, Any],
    library: CaseLibrary,
    *,
    min_quality: float = 0.5,
) -> bool:
    """
    Evaluate a solved case and conditionally store it in the case library.

    Parameters
    ----------
    case_id:
        Identifier to use for the retained case (must be unique in the library).
    problem:
        The problem dict (as produced by build_case.py).
    solution:
        The adapted solution object to store.
    outcome:
        Outcome metadata describing the result after solution application/review.
    library:
        CaseLibrary instance to store into.
    min_quality:
        Minimum quality score necessary to retain the case.

    Returns
    -------
    bool
        True if the case was stored (retained), False otherwise.
    """
    candidate = {
        "case_id": case_id,
        "problem": problem,
        "solution": solution,
        "outcome": outcome,
    }
    if should_retain(candidate, min_quality=min_quality):
        library.add(case_id, problem, solution, outcome)
        logger.info("Retained case '%s' (quality >= %s).", case_id, min_quality)
        return True

    logger.info("Did not retain case '%s' (quality < %s).", case_id, min_quality)
    return False


# Convenience wrapper: evaluate then record outcome (if the case exists)
def record_and_maybe_retain(
    case_id: str,
    outcome: Dict[str, Any],
    library: CaseLibrary,
    *,
    min_quality: float = 0.5,
    new_case_when_missing: bool = False,
    problem: Optional[Dict[str, Any]] = None,
    solution: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Record an outcome for a case and conditionally retain it. Returns True if retained.

    Parameters
    ----------
    - case_id, outcome, library: as above.
    - min_quality: threshold for retaining.
    - new_case_when_missing: if True and the case_id is not found, a new case will
      be created using `problem` and `solution` (both must be provided). If False,
      a missing case raises KeyError.
    """
    existing = library.get(case_id)
    if existing is None:
        if not new_case_when_missing:
            raise KeyError(f"Case '{case_id}' not found in the library.")
        if problem is None or solution is None:
            raise ValueError(
                "problem and solution must be provided to create a new case."
            )
        library.add(case_id, problem, solution, outcome)
        logger.info("Created and recorded outcome for new case '%s'.", case_id)
        return True if compute_quality_score(outcome) >= min_quality else False

    # Update outcome
    library.add(case_id, existing["problem"], existing["solution"], outcome)
    # Decide retention (if not already present, this will insert; since we updated above it's safe)
    return retain_case(
        case_id,
        existing["problem"],
        existing["solution"],
        outcome,
        library,
        min_quality=min_quality,
    )


if __name__ == "__main__":  # pragma: no cover - lightweight smoke test
    # Minimal demonstration when run as script.
    import sys

    from .case_library import CaseLibrary

    def _demo():
        lib = CaseLibrary()
        cid = "demo-001"
        problem = {
            "patient": {"age_years": 55},
            "clinical_data": {"diagnoses": [{"diagnosisString": "asthma"}]},
        }
        solution = {"plan": "inhaled bronchodilator PRN"}
        outcome = {"quality_score": 0.85, "reviewed_by": "demo"}
        print("Store path:", lib.db_path)
        print("Initial count:", lib.count())
        retained = retain_case(cid, problem, solution, outcome, lib, min_quality=0.5)
        print("Retained?:", retained)
        print("New count:", lib.count())
        # cleanup for demo purposes (best-effort)
        lib.remove(cid)

    try:
        _demo()
    except Exception as exc:  # pragma: no cover - demo error reporting
        print("Demo failed:", exc, file=sys.stderr)
        raise
