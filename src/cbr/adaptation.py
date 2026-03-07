"""
adaptation.py — Case adaptation / reuse logic for src.cbr

This module implements safe, deterministic adaptation strategies used in the
'Reuse' step of the CBR cycle. It intentionally keeps import-time side-effects
minimal and does not require heavy ML dependencies.

Provided strategies
-------------------
- null_adapt: return the retrieved solution unchanged (safe baseline).
- substitution_adapt: copy-and-substitute fields from the new problem into the
  retrieved solution using a declarative field mapping.
- rule_based_adapt: deterministic clinical-rule adaptation that adjusts
  severity, triage priority, and recommendations based on new-patient
  physiology compared to the retrieved case.  No GPU required.
- llm_adapt: builds a prompt and optionally calls a local MedGemma model to
  perform LLM-assisted adaptation on a GPU node.

Design notes
------------
- Field maps are intentionally simple and serializable (strings or small dicts).
- No network or GPU work is performed at import-time.
- rule_based_adapt is purely CPU-based and can run on a login node.
- llm_adapt imports torch/transformers lazily and MUST run on a GPU node.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

# Type for a field-map entry:
# - If str: dotted path into the `new_problem` to copy from.
# - If callable: a function(new_problem, retrieved_solution, retrieved_problem) -> value.
# - If dict: may contain keys:
#     - "from": dotted path or callable (as above)
#     - "transform": callable to post-process the resolved value
#     - "default": fallback value when resolution yields None
FieldMapSpec = Union[
    str,
    Callable[[dict, dict, Optional[dict]], Any],
    Dict[str, Any],
]
FieldMap = Dict[str, FieldMapSpec]


# ---------------------------
# Small dict-path utilities
# ---------------------------
def _resolve_path(obj: dict, dotted_path: str) -> Any:
    """
    Resolve a dotted path into a nested dict.

    Example:
        _resolve_path(case, "patient.age_years")
    Returns None if any intermediate is missing.
    """
    if not dotted_path:
        return None
    parts = dotted_path.split(".")
    cur: Any = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def _set_path(obj: dict, dotted_path: str, value: Any) -> None:
    """
    Set a value into a nested dict creating intermediate mappings as needed.

    Example:
        _set_path(sol, "treatment.recommended_dose_mg", 5.0)
    """
    parts = dotted_path.split(".")
    cur: dict = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ---------------------------
# Adaptation strategies
# ---------------------------
def null_adapt(
    new_problem: dict,
    retrieved_solution: dict,
    retrieved_problem: Optional[dict] = None,
) -> dict:
    """
    Identity adaptation — return a deep copy of the retrieved solution.

    Use this when the retrieved solution is already appropriate or when
    you want a safe baseline to compare against more sophisticated strategies.
    """
    return copy.deepcopy(retrieved_solution)


def substitution_adapt(
    new_problem: dict,
    retrieved_solution: dict,
    retrieved_problem: Optional[dict] = None,
    *,
    field_map: Optional[FieldMap] = None,
) -> dict:
    """
    Simple substitution adaptation.

    Parameters
    ----------
    new_problem:
        The target (unsolved) problem case produced by `build_case.py`.
    retrieved_solution:
        The solution object from the retrieved (similar) case.
    retrieved_problem:
        The original problem associated with the retrieved solution (optional).
    field_map:
        Mapping of `solution_field_path` -> spec. The spec may be:
          - a dotted path string into `new_problem` (e.g. "patient.weight_kg")
          - a callable: fn(new_problem, retrieved_solution, retrieved_problem) -> value
          - a dict with optional keys:
                "from": dotted path or callable (same semantics)
                "transform": callable(value) -> new_value
                "default": value to use when resolution yields None

    Returns
    -------
    adapted_solution : dict
        A deep-copy of `retrieved_solution` with substitutions applied.
    """
    adapted = copy.deepcopy(retrieved_solution)
    if not field_map:
        return adapted

    for sol_key, spec in field_map.items():
        value = None
        used_default = False

        # Resolve spec
        try:
            if isinstance(spec, str):
                value = _resolve_path(new_problem, spec)
            elif callable(spec):
                value = spec(new_problem, retrieved_solution, retrieved_problem)
            elif isinstance(spec, dict):
                src = spec.get("from")
                if isinstance(src, str):
                    value = _resolve_path(new_problem, src)
                elif callable(src):
                    value = src(new_problem, retrieved_solution, retrieved_problem)
                elif src is None:
                    # No explicit 'from' — leave value as None and rely on default
                    value = None
                else:
                    # Unknown 'from' type, try raw assign if present
                    value = src

                # Transform if requested
                transform = spec.get("transform")
                if transform and callable(transform):
                    value = transform(value)
                # Default handling
                if value is None and "default" in spec:
                    value = spec["default"]
                    used_default = True
            else:
                # Unknown spec type — skip
                continue
        except Exception:
            # Resolution errors shouldn't break adaptation; skip this mapping
            continue

        # If resolution gave no value and no default, skip substitution
        if value is None and not used_default:
            continue

        # Apply into adapted solution
        try:
            _set_path(adapted, sol_key, value)
        except Exception:
            # If setting fails, leave adapted unchanged for this key
            continue

    return adapted


# ---------------------------
# Rule-based clinical adaptation
# ---------------------------

logger = logging.getLogger(__name__)

# Severity flag weights used by rule_based_adapt to compute acuity scores.
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "hypotension": 1.5,
    "tachypnea": 1.0,
    "hypoxemia": 1.5,
    "hypothermia": 0.8,
    "fever": 0.6,
    "tachycardia": 0.8,
    "altered mental status": 1.5,
    "mechanical ventilation": 2.0,
    "vasopressor": 1.8,
    "dialysis": 1.5,
}


def _acuity_score(flags: List[str]) -> float:
    """Compute a weighted acuity score from a list of severity flag strings."""
    score = 0.0
    for flag in flags:
        flag_lower = flag.lower()
        for keyword, weight in _SEVERITY_WEIGHTS.items():
            if keyword in flag_lower:
                score += weight
                break
        else:
            score += 0.5  # unknown flag gets a small base weight
    return round(score, 2)


def _triage_priority(acuity: float) -> str:
    """Map an acuity score to a triage priority label."""
    if acuity >= 6.0:
        return "High"
    if acuity >= 3.0:
        return "Medium"
    return "Low"


def _compare_vitals(
    new_vitals: Dict[str, Any], retrieved_vitals: Dict[str, Any]
) -> List[str]:
    """Generate human-readable comparison notes between two acute-physiology dicts."""
    notes: List[str] = []

    # MAP comparison
    new_map = (new_vitals.get("map") or {}).get("value")
    ret_map = (retrieved_vitals.get("map") or {}).get("value")
    if new_map is not None and ret_map is not None:
        diff = new_map - ret_map
        if abs(diff) > 5:
            direction = "higher" if diff > 0 else "lower"
            notes.append(f"MAP is {abs(diff):.0f} mmHg {direction} than retrieved case")

    # Heart rate
    new_hr = (new_vitals.get("heart_rate") or {}).get("value")
    ret_hr = (retrieved_vitals.get("heart_rate") or {}).get("value")
    if new_hr is not None and ret_hr is not None:
        diff = new_hr - ret_hr
        if abs(diff) > 10:
            direction = "higher" if diff > 0 else "lower"
            notes.append(f"Heart rate is {abs(diff):.0f} bpm {direction}")

    # SpO2
    new_spo2 = (new_vitals.get("oxygenation") or {}).get("spo2_min")
    ret_spo2 = (retrieved_vitals.get("oxygenation") or {}).get("spo2_min")
    if new_spo2 is not None and ret_spo2 is not None:
        diff = new_spo2 - ret_spo2
        if abs(diff) > 3:
            direction = "better" if diff > 0 else "worse"
            notes.append(f"SpO2 is {abs(diff):.0f}% {direction}")

    # GCS
    new_gcs = (new_vitals.get("gcs") or {}).get("total")
    ret_gcs = (retrieved_vitals.get("gcs") or {}).get("total")
    if new_gcs is not None and ret_gcs is not None and new_gcs > 0 and ret_gcs > 0:
        diff = new_gcs - ret_gcs
        if abs(diff) >= 2:
            direction = "higher (better)" if diff > 0 else "lower (worse)"
            notes.append(f"GCS is {abs(diff)} points {direction}")

    return notes


def rule_based_adapt(
    new_problem: Dict[str, Any],
    retrieved_solution: Dict[str, Any],
    retrieved_problem: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic rule-based adaptation of a retrieved triage case.

    Compares the new patient's physiology and demographics against the
    retrieved case and produces an adapted solution dict containing:

    - adapted triage priority (High / Medium / Low)
    - new patient acuity score vs. retrieved case acuity score
    - comparison notes highlighting key physiological differences
    - recommended adjustments to organ-support and monitoring
    - the original retrieved solution (unchanged) for reference

    This function is CPU-only and safe to run on a login node.

    Parameters
    ----------
    new_problem : dict
        The target (unsolved) triage case produced by ``build_triage_case.py``.
    retrieved_solution : dict
        The full ``source_case`` dict from the retrieved similar case.
    retrieved_problem : dict, optional
        Ignored for rule-based; kept for interface consistency.

    Returns
    -------
    dict
        An adapted-solution dict with keys:
        ``adapted_priority``, ``new_acuity``, ``retrieved_acuity``,
        ``acuity_delta``, ``comparison_notes``, ``recommended_adjustments``,
        ``adapted_at``, ``strategy``, ``original_retrieved_solution``.
    """
    # -- Extract severity flags --
    new_flags: List[str] = (
        new_problem.get("triage_context", {}).get("severity_flags", [])
    )
    ret_flags: List[str] = (
        retrieved_solution.get("triage_context", {}).get("severity_flags", [])
    )

    new_acuity = _acuity_score(new_flags)
    ret_acuity = _acuity_score(ret_flags)
    acuity_delta = round(new_acuity - ret_acuity, 2)

    # -- Triage priority --
    adapted_priority = _triage_priority(new_acuity)

    # -- Vitals comparison --
    new_vitals = new_problem.get("acute_physiology_24h", {})
    ret_vitals = retrieved_solution.get("acute_physiology_24h", {})
    comparison_notes = _compare_vitals(new_vitals, ret_vitals)

    # -- Organ-support recommendations --
    adjustments: List[str] = []

    new_organ = new_problem.get("organ_support", {})
    ret_organ = retrieved_solution.get("organ_support", {})

    # Check if the new patient needs support that the retrieved patient had
    for support_type in ("mechanical_ventilation", "vasopressor_use", "dialysis"):
        new_flag = _get_organ_flag(new_organ, support_type)
        ret_flag = _get_organ_flag(ret_organ, support_type)
        label = support_type.replace("_", " ")

        if new_flag and not ret_flag:
            adjustments.append(
                f"New patient requires {label} — retrieved case did not. "
                "Escalate monitoring."
            )
        elif ret_flag and not new_flag:
            adjustments.append(
                f"Retrieved case required {label} — new patient does not currently. "
                "Monitor closely for deterioration."
            )

    # Age-based notes
    new_age = new_problem.get("demographics", {}).get("age")
    ret_age = retrieved_solution.get("demographics", {}).get("age")
    if new_age is not None and ret_age is not None:
        try:
            age_diff = int(new_age) - int(ret_age)
            if abs(age_diff) > 10:
                direction = "older" if age_diff > 0 else "younger"
                adjustments.append(
                    f"New patient is {abs(age_diff)} years {direction} — "
                    "consider age-adjusted thresholds for interventions."
                )
        except (ValueError, TypeError):
            pass

    # Comorbidity notes
    new_comorb = new_problem.get("comorbidities", {}).get("flags", {})
    ret_comorb = retrieved_solution.get("comorbidities", {}).get("flags", {})
    extra_comorbs = [
        k for k, v in new_comorb.items() if v and not ret_comorb.get(k)
    ]
    if extra_comorbs:
        adjustments.append(
            f"New patient has additional comorbidities not in retrieved case: "
            f"{', '.join(extra_comorbs)}. Adjust treatment plan accordingly."
        )

    # Retrieved case outcome-based warnings
    ret_mortality = (
        retrieved_solution.get("icu_los", {}).get("hospital_mortality")
    )
    if ret_mortality:
        adjustments.append(
            "WARNING: Retrieved similar case had hospital mortality. "
            "Treat new patient with heightened vigilance."
        )

    return {
        "adapted_priority": adapted_priority,
        "new_acuity": new_acuity,
        "retrieved_acuity": ret_acuity,
        "acuity_delta": acuity_delta,
        "comparison_notes": comparison_notes,
        "recommended_adjustments": adjustments,
        "adapted_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "rule_based",
        "original_retrieved_solution": copy.deepcopy(retrieved_solution),
    }


def _get_organ_flag(organ_dict: Dict[str, Any], key: str) -> bool:
    """Safely extract an organ-support boolean flag."""
    entry = organ_dict.get(key, {})
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, dict):
        return bool(entry.get("flag"))
    return False


# ---------------------------
# LLM-assisted adaptation (placeholder)
# ---------------------------
def build_llm_prompt(
    new_problem: dict,
    retrieved_problem: dict,
    retrieved_solution: dict,
    *,
    instructions: Optional[str] = None,
) -> str:
    """
    Compose a human-readable prompt for an LLM to adapt a retrieved solution.

    The prompt includes:
      - short instructions (optional)
      - serialized new problem
      - serialized retrieved problem + solution
      - explicit output format request (JSON)

    The returned prompt is safe to send to an LLM backend that accepts a
    plain-text prompt.
    """
    instructions = instructions or (
        "You are a clinical assistant. Given a new patient case and a similar"
        " retrieved case with its recommended solution, produce an adapted"
        " solution for the new patient. Be conservative and highlight any"
        " assumptions made. Return the adapted solution as a JSON object only."
    )

    # Use compact JSON blobs to keep prompts concise
    np_json = json.dumps(new_problem, indent=2, ensure_ascii=False)
    rp_json = json.dumps(retrieved_problem or {}, indent=2, ensure_ascii=False)
    rs_json = json.dumps(retrieved_solution or {}, indent=2, ensure_ascii=False)

    prompt = (
        f"{instructions}\n\n"
        "=== New patient problem ===\n"
        f"{np_json}\n\n"
        "=== Retrieved (similar) problem ===\n"
        f"{rp_json}\n\n"
        "=== Retrieved solution ===\n"
        f"{rs_json}\n\n"
        "=== Instructions for the adapted solution ===\n"
        "Return a JSON object describing the adapted solution. Include any fields\n"
        "that were changed and a short 'rationale' field explaining why they were\n"
        "changed. Do not include any additional text outside the JSON object."
    )
    return prompt


def llm_adapt(
    new_problem: dict,
    retrieved_solution: dict,
    retrieved_problem: Optional[dict] = None,
    *,
    model_hint: Optional[str] = None,
    run_local: bool = False,
) -> dict:
    """
    LLM-assisted adaptation using a local MedGemma model.

    When ``run_local=True`` and a ``model_hint`` path is provided, this function
    lazily imports ``torch`` and ``transformers``, loads the model, and runs
    inference.  **This MUST be executed on a GPU compute node.**

    When ``run_local=False`` (default), it returns a dict containing the
    generated prompt and instructions for manual execution — safe for login
    nodes.

    Parameters
    ----------
    new_problem : dict
        The new (unsolved) triage case.
    retrieved_solution : dict
        The source_case from the retrieved similar case.
    retrieved_problem : dict, optional
        The original problem associated with the retrieved solution.
    model_hint : str, optional
        Filesystem path to local MedGemma weights (e.g.
        ``/N/scratch/alikh/models/google--medgemma-27b-it``).
    run_local : bool
        If True, load the model and run inference locally on a GPU node.

    Returns
    -------
    dict
        Keys: ``adapted_solution`` (parsed JSON or raw text), ``prompt``,
        ``model_used``, ``strategy``, ``adapted_at``.

    Raises
    ------
    NotImplementedError
        When ``run_local=False`` and the caller expects actual inference.
    RuntimeError
        When ``run_local=True`` but no GPU is available or model loading fails.
    """
    prompt = build_llm_prompt(new_problem, retrieved_problem or {}, retrieved_solution)

    if not run_local:
        # Return the prompt so the caller can route it however they wish
        return {
            "adapted_solution": None,
            "prompt": prompt,
            "model_used": model_hint or "none (prompt-only mode)",
            "strategy": "llm_prompt_only",
            "adapted_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "run_local=False — no inference was performed. Send the prompt "
                "to your chosen LLM backend, or re-call with run_local=True "
                "on a GPU compute node."
            ),
        }

    # --- Local GPU inference path ---
    import os

    model_path = model_hint or os.getenv(
        "MODEL_PATH", "/N/scratch/alikh/models/google--medgemma-27b-it"
    )

    logger.info("llm_adapt: loading model from %s (run_local=True)", model_path)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            f"torch/transformers not importable — ensure you are on a GPU node "
            f"with the correct venv activated: {e}"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. llm_adapt with run_local=True must run on a "
            "GPU compute node (use srun or sbatch)."
        )

    device = "cuda"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
        )

    # Decode only the generated portion (skip the prompt tokens)
    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # Try to parse as JSON; fall back to raw text
    adapted_solution: Any
    try:
        adapted_solution = json.loads(generated_text)
    except json.JSONDecodeError:
        # Attempt to extract JSON block from markdown fences
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", generated_text, re.DOTALL)
        if match:
            try:
                adapted_solution = json.loads(match.group(1))
            except json.JSONDecodeError:
                adapted_solution = {"raw_text": generated_text}
        else:
            adapted_solution = {"raw_text": generated_text}

    # Clean up GPU memory
    del model, tokenizer, inputs, output_ids
    torch.cuda.empty_cache()

    return {
        "adapted_solution": adapted_solution,
        "prompt": prompt,
        "model_used": model_path,
        "strategy": "llm_local",
        "adapted_at": datetime.now(timezone.utc).isoformat(),
    }
