"""
retrieval.py — KG-backed similarity computation and case retrieval for src.cbr

This module implements the **Retrieve** step of the CBR cycle by querying the
Oxigraph RDF Knowledge Graph to find candidate patients and then ranking them
using a composite similarity score.

The patientunitstayid is used as the canonical case_id.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    limit: int = 50,
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
    store, keyword_lower, *, age=None, age_tolerance=15, limit=50
):
    """Search diagnosis strings directly."""
    age_filter = ""
    if age is not None:
        low, high = max(0, age - age_tolerance), age + age_tolerance
        age_filter = (
            f"?pat eicu:age ?patAge . FILTER(?patAge >= {low} && ?patAge <= {high})"
        )

    sparql = f"""
{_SPARQL_PREFIXES}
SELECT DISTINCT ?pat ?patAge ?patGender ?diagStr
WHERE {{
    ?diag rdf:type eicu:Diagnosis ;
          eicu:diagnosisString ?diagStr .
    FILTER(CONTAINS(LCASE(STR(?diagStr)), "{keyword_lower}"))
    ?pat rdf:type eicu:Patient ;
         eicu:hasDiagnosis ?diag .
    {age_filter}
    OPTIONAL {{ ?pat eicu:age ?patAge }}
    OPTIONAL {{ ?pat eicu:gender ?patGender }}
}}
LIMIT {limit}
"""
    results = list(store.query(sparql))
    candidates = []
    seen = set()
    for row in results:
        pat_uri = str(row["pat"].value)
        stayid = pat_uri.rsplit("/", 1)[-1]
        if stayid in seen:
            continue
        seen.add(stayid)
        candidates.append(
            {
                "stayid": stayid,
                "age": int(row["patAge"].value) if row["patAge"] else None,
                "gender": str(row["patGender"].value) if row["patGender"] else None,
                "matched_problem": str(row["diagStr"].value) if row["diagStr"] else "",
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Similarity Scoring logic
# ---------------------------------------------------------------------------


def _numeric_closeness(
    a: Optional[float], b: Optional[float], max_diff: float
) -> float:
    """1.0 when equal, linearly declining to 0.0 at max_diff."""
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(float(a) - float(b)), max_diff) / max_diff)


def compute_similarity(
    query_age: int,
    query_problem: str,
    candidate_case: Dict[str, Any],
    *,
    w_diag: float = 0.30,
    w_age: float = 0.40,
    w_completeness: float = 0.30,
) -> SimilarityResult:
    """Score a candidate case against the query."""
    # 1. Age Score
    # Triage cases store age in 'demographics' -> 'age'
    # Fallback to 'patient' -> 'demographics' -> 'age' for old generic cases
    cand_age = candidate_case.get("demographics", {}).get("age")
    if cand_age is None:
        cand_age = candidate_case.get("patient", {}).get("demographics", {}).get("age")
    
    # Ensure numerical comparison
    try:
        if isinstance(cand_age, str) and ">" in cand_age:
            cand_age = 90
        else:
            cand_age = float(cand_age)
    except (ValueError, TypeError):
        cand_age = None

    age_sim = _numeric_closeness(float(query_age), cand_age, max_diff=30.0)

    # 2. Diagnosis/Problem Match (Checking if problem appears in comorbidities)
    # Triage cases have 'comorbidities' list
    cand_comorbs = candidate_case.get("comorbidities", [])
    if isinstance(cand_comorbs, list):
        comorb_text = " ".join([str(c).lower() for c in cand_comorbs])
    else:
        comorb_text = ""
    
    query_kw = query_problem.lower().strip()
    # If the problem is "sepsis" (acute), it might not be in comorbidities.
    # But if it is "copd" (chronic), it likely is.
    # We assign partial score if found, but don't penalize too hard if missing (KG already matched it implicitly)
    diag_match = 1.0 if query_kw in comorb_text else 0.5

    # 3. Completeness/Acuity (Prefer cases with more active data)
    # Count active organ supports or populated sections
    organ_support = candidate_case.get("organ_support", {})
    active_supports = sum(1 for k, v in organ_support.items() if (isinstance(v, bool) and v) or (isinstance(v, dict) and v.get("flag")))
    
    completeness = min(1.0, active_supports / 3.0) # Bonus for rich cases (vent/dialysis/pressors)
    if not completeness and candidate_case.get("key_labs_24h"):
        completeness = 0.2  # Base score if labs exist

    total = w_diag * diag_match + w_age * age_sim + w_completeness * completeness
    total = max(0.0, min(1.0, total))

    # ID retrieval
    stayid = str(candidate_case.get("metadata", {}).get("patientunitstayid", 
                 candidate_case.get("metadata", {}).get("patient_unit_stay_id", "?")))

    # Extract useful display info
    triage_context = candidate_case.get("triage_context", {})
    severity_flags = triage_context.get("severity_flags", [])

    return SimilarityResult(
        case_id=stayid,
        score=round(total, 4),
        patient_age=int(cand_age) if cand_age else None,
        patient_gender=candidate_case.get("demographics", {}).get("gender"),
        primary_problem=query_problem,
        details={
            "diagnosis_match": round(diag_match, 2),
            "age_similarity": round(age_sim, 2),
            "case_completeness": round(completeness, 2),
            "active_organ_support": [k for k, v in organ_support.items() if (isinstance(v, bool) and v) or (isinstance(v, dict) and v.get("flag"))],
            "severity_flags": severity_flags
        },
        source_case=candidate_case
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
) -> List[SimilarityResult]:
    """Retrieve top-k similar patient cases using KG discovery."""
    if not name or not problem or age is None:
        raise ValueError(
            "Mandatory fields 'name', 'age', and 'problem' must be provided."
        )

    logger.info("CBR Retrieve: name=%s, age=%d, problem=%s", name, age, problem)

    # 1. Query KG for candidate stayids
    store = _get_store()
    candidates = _find_candidates_by_problem(
        store, problem, age=age, age_tolerance=age_tolerance
    )

    if not candidates:
        return []

    # 2. Import build_triage_case dynamically
    try:
        from src.inference.build_triage_case import build_triage_case
    except ImportError:
        # Fallback if specific triage import fails (though it should be there)
        from src.inference.build_case import build_case as build_triage_case

    # 3. Construct and Score
    results: List[SimilarityResult] = []
    for cand in candidates:
        stayid = int(cand["stayid"])
        try:
            # Build full patient case from SQLite (Triage-optimized)
            case_json = build_triage_case(stayid)
            # Compare against original query
            sim = compute_similarity(age, problem, case_json)
            results.append(sim)
        except Exception as e:
            logger.warning("Skipping stayid %s: %s", stayid, e)
            continue

    # 4. Rank by score
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
