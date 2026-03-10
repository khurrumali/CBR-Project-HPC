"""
retrieval.py — KG-backed similarity computation and case retrieval for src.cbr

This module implements the **Retrieve** step of the CBR cycle by querying the
Oxigraph RDF Knowledge Graph to find candidate patients and then ranking them
using a composite similarity score.

The patientunitstayid is used as the canonical case_id.

Similarity dimensions (v2):
  - Diagnosis token overlap   (0.28) — Jaccard over tokenized diagnosis text
  - Age closeness             (0.25) — linear decay over ±30 years
  - Comorbidity overlap       (0.20) — Jaccard over active comorbidity flags
  - Severity alignment        (0.15) — candidate severity flag richness
  - Gender match              (0.07) — exact match bonus
  - APACHE score proximity    (0.05) — closeness over ±30 APACHE points
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy dependency resolvers
# ---------------------------------------------------------------------------


def _get_store():
    """Open the Oxigraph store from the standard env path."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    from pyoxigraph import Store

    store_path = os.getenv("OXIGRAPH_STORE_PATH", "/N/scratch/alikh/oxigraph_store")
    if not os.path.isdir(store_path):
        raise FileNotFoundError(
            f"Oxigraph store not found at {store_path}. "
            "Run  python src/knowledge_graph/build_kg.py  first."
        )
    return Store(store_path)


# Standard SPARQL prefix block matching the project's rdf_schema.py
_SPARQL_PREFIXES = """
PREFIX eicu: <http://eicu.mit.edu/ontology/>
PREFIX data: <http://eicu.mit.edu/data/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
""".strip()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SimilarityResult:
    """A candidate case with its similarity score and breakdown."""

    case_id: str  # patientunitstayid (string)
    score: float  # 0.0 → 1.0
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    primary_problem: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    source_case: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SPARQL-based candidate discovery
# ---------------------------------------------------------------------------


def _find_candidates_by_problem(
    store,
    problem_keyword: str,
    *,
    age: Optional[int] = None,
    age_tolerance: int = 15,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Query the KG for patients matching a clinical problem keyword via SKOS concepts."""
    keyword_lower = problem_keyword.strip().lower()

    # Build age filter if provided
    age_filter = ""
    if age is not None:
        low = max(0, age - age_tolerance)
        high = age + age_tolerance
        age_filter = f"""
        ?pat eicu:age ?patAge .
        FILTER(?patAge >= {low} && ?patAge <= {high})
        """

    # Query: SKOS concept lookup -> diagnosis -> patient
    sparql = f"""
{_SPARQL_PREFIXES}
SELECT DISTINCT ?pat ?patAge ?patGender ?prefLabel
WHERE {{
    # 1. Match SKOS concepts by label
    ?concept rdf:type skos:Concept ;
             skos:prefLabel ?prefLabel .
    FILTER(CONTAINS(LCASE(STR(?prefLabel)), "{keyword_lower}"))

    # 2. Match diagnoses to those concepts
    ?diag rdf:type eicu:Diagnosis ;
          eicu:problem ?diagProblem .
    FILTER(LCASE(STR(?diagProblem)) = LCASE(STR(?prefLabel)))

    # 3. Join to patients
    ?pat rdf:type eicu:Patient ;
         eicu:hasDiagnosis ?diag .

    {age_filter}

    OPTIONAL {{ ?pat eicu:age ?patAge }}
    OPTIONAL {{ ?pat eicu:gender ?patGender }}
}}
LIMIT {limit}
"""
    results = list(store.query(sparql))
    candidates: List[Dict[str, Any]] = []
    seen_stayids: set = set()

    for row in results:
        try:
            pat_uri = str(row["pat"].value)
            stayid = pat_uri.rsplit("/", 1)[-1]
            if stayid in seen_stayids:
                continue
            seen_stayids.add(stayid)

            p_age = int(row["patAge"].value) if row["patAge"] else None
            p_gender = str(row["patGender"].value) if row["patGender"] else None
            p_problem = str(row["prefLabel"].value) if row["prefLabel"] else ""

            candidates.append(
                {
                    "stayid": stayid,
                    "age": p_age,
                    "gender": p_gender,
                    "matched_problem": p_problem,
                }
            )
        except Exception:
            continue

    # Fallback: direct diagnosis string search if SKOS returns nothing
    if not candidates:
        candidates = _fallback_diagnosis_search(
            store, keyword_lower, age=age, age_tolerance=age_tolerance, limit=limit
        )

    return candidates


def _fallback_diagnosis_search(
    store, keyword_lower, *, age=None, age_tolerance=15, limit=200
):
    """Search full diagnosisString directly when problem-level match fails."""
    age_filter = ""
    if age is not None:
        low, high = max(0, age - age_tolerance), age + age_tolerance
        age_filter = f"FILTER(!BOUND(?patAge) || (?patAge >= {low} && ?patAge <= {high}))"

    sparql = f"""
{_SPARQL_PREFIXES}
SELECT DISTINCT ?pat ?patAge ?patGender ?diagStr
WHERE {{
    ?diag rdf:type eicu:Diagnosis ;
          eicu:diagnosisString ?diagStr .
    FILTER(CONTAINS(LCASE(STR(?diagStr)), "{keyword_lower}"))
    ?pat rdf:type eicu:Patient ;
         eicu:hasDiagnosis ?diag .
    OPTIONAL {{ ?pat eicu:age ?patAge . }}
    OPTIONAL {{ ?pat eicu:gender ?patGender . }}
    {age_filter}
}}
LIMIT {limit}
"""
    results = list(store.query(sparql))
    candidates = []
    seen = set()
    for row in results:
        try:
            pat_uri = str(row["pat"].value)
            stayid = pat_uri.rsplit("/", 1)[-1]
            if stayid in seen:
                continue
            seen.add(stayid)
            try:
                p_age = int(row["patAge"].value) if row["patAge"] is not None else None
            except Exception:
                p_age = None
            try:
                p_gender = str(row["patGender"].value) if row["patGender"] is not None else None
            except Exception:
                p_gender = None
            try:
                p_problem = str(row["diagStr"].value) if row["diagStr"] is not None else ""
            except Exception:
                p_problem = ""
            candidates.append(
                {
                    "stayid": stayid,
                    "age": p_age,
                    "gender": p_gender,
                    "matched_problem": p_problem,
                }
            )
        except Exception:
            continue
    return candidates


# ---------------------------------------------------------------------------
# Case Library candidate source
# ---------------------------------------------------------------------------


def _retrieve_from_case_library(problem_keyword: str) -> List[Dict[str, Any]]:
    """
    Search the CaseLibrary for previously solved cases matching the problem keyword.
    Returns hydrated case dicts tagged with '_from_library': True.
    """
    try:
        from src.cbr.case_library import CaseLibrary
        lib = CaseLibrary()
        if lib.count() == 0:
            return []

        keyword_lower = problem_keyword.strip().lower()
        matched = []
        for case in lib.all_cases():
            problem = case.get("problem") or {}
            # Match against stored problem text fields
            searchable = " ".join([
                str(problem.get("problem", "")),
                str(problem.get("admitting_diagnosis", "")),
                str(problem.get("primary_diagnosis_string", "")),
            ]).lower()
            if keyword_lower in searchable:
                # Reconstruct a candidate dict from stored problem representation
                solution = case.get("solution") or {}
                candidate = {**solution, "_from_library": True, "_case_id": case["case_id"]}
                matched.append(candidate)
        return matched
    except Exception as exc:
        logger.debug("CaseLibrary lookup skipped: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _numeric_closeness(
    a: Optional[float], b: Optional[float], max_diff: float
) -> float:
    """1.0 when equal, linearly declining to 0.0 at max_diff."""
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(float(a) - float(b)), max_diff) / max_diff)


def _token_overlap(a: str, b: str) -> float:
    """
    Jaccard token overlap between two clinical text strings.
    Strips punctuation, lowercases, splits on whitespace.
    Returns 0.0 if either string is empty after tokenization.
    """
    def tokenize(s: str) -> Set[str]:
        return set(re.sub(r"[^\w\s]", " ", s.lower()).split())

    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _comorbidity_jaccard(
    query_flags: Optional[Set[str]],
    candidate_case: Dict[str, Any],
) -> float:
    """
    Jaccard overlap between query comorbidity flags and candidate flags.

    - If query_flags is None (unknown), returns 0.0 (no contribution).
    - If both sets are empty (both healthy), returns 1.0 (perfect match).
    - Otherwise returns intersection / union.
    """
    if query_flags is None:
        return 0.0  # query comorbidities unknown — skip this dimension

    cand_comorbs = candidate_case.get("comorbidities", {})
    if isinstance(cand_comorbs, dict):
        cand_flags = {k for k, v in cand_comorbs.get("flags", {}).items() if v}
    else:
        cand_flags = set()

    if not query_flags and not cand_flags:
        return 1.0  # both healthy — perfect comorbidity match
    if not query_flags or not cand_flags:
        return 0.0
    intersection = query_flags & cand_flags
    union = query_flags | cand_flags
    return len(intersection) / len(union)


def _severity_richness(candidate_case: Dict[str, Any]) -> float:
    """
    Normalize candidate severity flag count to [0, 1].
    More severity flags → richer, more acute case → higher score.
    Uses triage_context.severity_count if available, else counts organ supports.
    Scale: 0 flags = 0.0, 4+ flags = 1.0.
    """
    triage = candidate_case.get("triage_context", {})
    count = triage.get("severity_count")
    if count is not None:
        return min(1.0, count / 4.0)

    # Fallback: count active organ supports
    organ_support = candidate_case.get("organ_support", {})
    active = sum(
        1 for v in organ_support.values()
        if (isinstance(v, bool) and v) or (isinstance(v, dict) and v.get("flag"))
    )
    return min(1.0, active / 3.0)


# ---------------------------------------------------------------------------
# Similarity Scoring logic
# ---------------------------------------------------------------------------


def compute_similarity(
    query_age: int,
    query_problem: str,
    candidate_case: Dict[str, Any],
    *,
    query_gender: Optional[str] = None,
    query_comorbidity_flags: Optional[Set[str]] = None,
    w_diag: float = 0.28,
    w_age: float = 0.25,
    w_comorbidities: float = 0.20,
    w_severity: float = 0.15,
    w_gender: float = 0.07,
    w_apache: float = 0.05,
) -> SimilarityResult:
    """
    Score a candidate case against the query using six clinical dimensions.

    Parameters
    ----------
    query_age : int
        Age of the query patient.
    query_problem : str
        Primary clinical problem/diagnosis string for the query patient.
    candidate_case : dict
        Full triage case JSON from build_triage_case().
    query_gender : str, optional
        Gender of the query patient ('Male'/'Female'). If None, gender dim is neutral.
    query_comorbidity_flags : set of str, optional
        Active comorbidity flag names for the query patient (e.g. {'chf', 'ckd'}).
        If None, comorbidity dimension contributes 0.
    """
    demo = candidate_case.get("demographics", {})

    # ── 1. Age Score ──────────────────────────────────────────────────────────
    cand_age = demo.get("age")
    if cand_age is None:
        # Legacy case structure fallback
        cand_age = candidate_case.get("patient", {}).get("demographics", {}).get("age")
    try:
        if isinstance(cand_age, str) and ">" in cand_age:
            cand_age = 90
        else:
            cand_age = float(cand_age)
    except (ValueError, TypeError):
        cand_age = None

    age_sim = _numeric_closeness(float(query_age), cand_age, max_diff=30.0)

    # ── 2. Diagnosis Token-Overlap Score ──────────────────────────────────────
    # Build a candidate diagnosis text blob from all diagnosis fields
    cand_comorbs = candidate_case.get("comorbidities", {})
    if isinstance(cand_comorbs, dict):
        flag_names = [k for k, v in cand_comorbs.get("flags", {}).items() if v]
        evidence_vals = []
        for ev_list in cand_comorbs.get("evidence", {}).values():
            evidence_vals.extend(ev_list)
        comorb_text = " ".join(flag_names + evidence_vals)
    elif isinstance(cand_comorbs, list):
        comorb_text = " ".join(str(c) for c in cand_comorbs)
    else:
        comorb_text = ""

    dx_text = " ".join(filter(None, [
        str(demo.get("primary_diagnosis_string") or ""),
        str(demo.get("admitting_diagnosis") or ""),
        str(demo.get("apache_admission_dx") or ""),
    ]))
    combined_text = f"{comorb_text} {dx_text}".strip()

    # Token-overlap Jaccard between query problem and candidate diagnosis text
    diag_sim = _token_overlap(query_problem, combined_text)
    # Ensure a meaningful floor when keyword is literally present (substring match)
    if query_problem.lower().strip() in combined_text.lower():
        diag_sim = max(diag_sim, 0.60)

    # ── 3. Comorbidity Jaccard ────────────────────────────────────────────────
    comorb_sim = _comorbidity_jaccard(query_comorbidity_flags, candidate_case)

    # ── 4. Severity Richness ──────────────────────────────────────────────────
    severity_sim = _severity_richness(candidate_case)

    # ── 5. Gender Match ───────────────────────────────────────────────────────
    cand_gender = demo.get("gender") or candidate_case.get("patient", {}).get("demographics", {}).get("gender")
    if query_gender is None or cand_gender is None:
        gender_sim = 0.5  # unknown — neutral
    elif query_gender.lower() == cand_gender.lower():
        gender_sim = 1.0
    else:
        gender_sim = 0.0

    # ── 6. APACHE Score Proximity ─────────────────────────────────────────────
    cand_apache = candidate_case.get("apache", {}).get("apache_score")
    # Only score this dimension if the candidate has an APACHE score
    if cand_apache is not None:
        # We don't know query APACHE, so use a fixed midpoint (25) as neutral reference
        # This rewards candidates near typical ICU acuity range rather than penalizing
        # Treat as richness signal: higher APACHE = higher acuity documented case
        apache_sim = min(1.0, float(cand_apache) / 40.0)  # normalize: 40 = severe
    else:
        apache_sim = 0.0

    # ── Weighted Sum ──────────────────────────────────────────────────────────
    # When comorbidity flags are unknown, redistribute its weight to diagnosis
    if query_comorbidity_flags is None:
        effective_w_diag = w_diag + w_comorbidities
        effective_w_comorb = 0.0
    else:
        effective_w_diag = w_diag
        effective_w_comorb = w_comorbidities

    total = (
        effective_w_diag * diag_sim
        + w_age * age_sim
        + effective_w_comorb * comorb_sim
        + w_severity * severity_sim
        + w_gender * gender_sim
        + w_apache * apache_sim
    )
    total = round(max(0.0, min(1.0, total)), 4)

    # ── Extract display metadata ──────────────────────────────────────────────
    stayid = str(
        candidate_case.get("metadata", {}).get("patientunitstayid")
        or candidate_case.get("metadata", {}).get("patient_unit_stay_id")
        or "?"
    )
    organ_support = candidate_case.get("organ_support", {})
    triage_context = candidate_case.get("triage_context", {})

    return SimilarityResult(
        case_id=stayid,
        score=total,
        patient_age=int(cand_age) if cand_age is not None else None,
        patient_gender=cand_gender,
        primary_problem=query_problem,
        details={
            "diagnosis_overlap": round(diag_sim, 4),
            "age_similarity": round(age_sim, 4),
            "comorbidity_jaccard": round(comorb_sim, 4),
            "severity_richness": round(severity_sim, 4),
            "gender_match": round(gender_sim, 4),
            "apache_proximity": round(apache_sim, 4),
            "apache_score": cand_apache,
            "severity_flags": triage_context.get("severity_flags", []),
            "active_organ_support": [
                k for k, v in organ_support.items()
                if (isinstance(v, bool) and v) or (isinstance(v, dict) and v.get("flag"))
            ],
        },
        source_case=candidate_case,
    )


# ---------------------------------------------------------------------------
# Main Retrieval Function
# ---------------------------------------------------------------------------


def retrieve(
    *,
    name: str,
    age: int,
    problem: str,
    top_k: int = 5,
    age_tolerance: int = 15,
    gender: Optional[str] = None,
    comorbidity_flags: Optional[Set[str]] = None,
) -> List[SimilarityResult]:
    """
    Retrieve top-k similar patient cases using KG discovery + CaseLibrary.

    Parameters
    ----------
    name : str
        Query patient name (used for logging only).
    age : int
        Query patient age.
    problem : str
        Primary clinical problem/diagnosis string.
    top_k : int
        Number of top results to return.
    age_tolerance : int
        ±years for the SPARQL age pre-filter.
    gender : str, optional
        Query patient gender ('Male'/'Female'). Passed to compute_similarity.
    comorbidity_flags : set of str, optional
        Active comorbidity flag names for the query patient (e.g. {'chf', 'ckd'}).
        When provided, enables Jaccard comorbidity scoring.
    """
    if not name or not problem or age is None:
        raise ValueError(
            "Mandatory fields 'name', 'age', and 'problem' must be provided."
        )

    logger.info("CBR Retrieve: name=%s, age=%d, problem=%s", name, age, problem)

    # 1. Query KG for candidate stayids
    store = _get_store()
    kg_candidates = _find_candidates_by_problem(
        store, problem, age=age, age_tolerance=age_tolerance
    )

    # 2. Import build_triage_case dynamically
    try:
        from src.inference.build_triage_case import build_triage_case
    except ImportError:
        from src.inference.build_case import build_case as build_triage_case

    # 3. Hydrate KG candidates from SQLite
    results: List[SimilarityResult] = []
    hydrated_ids: set = set()

    for cand in kg_candidates:
        stayid = int(cand["stayid"])
        if stayid in hydrated_ids:
            continue
        try:
            case_json = build_triage_case(stayid)
            sim = compute_similarity(
                age, problem, case_json,
                query_gender=gender,
                query_comorbidity_flags=comorbidity_flags,
            )
            results.append(sim)
            hydrated_ids.add(stayid)
        except Exception as e:
            logger.warning("Skipping KG stayid %s: %s", stayid, e)
            continue

    # 4. Merge CaseLibrary candidates (previously solved cases)
    lib_candidates = _retrieve_from_case_library(problem)
    lib_scored = 0
    for lib_case in lib_candidates:
        case_id = str(lib_case.get("_case_id", lib_case.get("metadata", {}).get("patient_unit_stay_id", "?")))
        # Avoid rescoring a case already retrieved from KG
        try:
            numeric_id = int(case_id)
            if numeric_id in hydrated_ids:
                continue
        except (ValueError, TypeError):
            pass
        try:
            sim = compute_similarity(
                age, problem, lib_case,
                query_gender=gender,
                query_comorbidity_flags=comorbidity_flags,
            )
            results.append(sim)
            lib_scored += 1
        except Exception as e:
            logger.warning("Skipping CaseLibrary case %s: %s", case_id, e)
            continue

    if lib_scored:
        logger.info("CaseLibrary contributed %d additional candidates", lib_scored)

    # 5. Rank by score and return top-k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
