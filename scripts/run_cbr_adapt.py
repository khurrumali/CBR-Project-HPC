#!/usr/bin/env python3
"""
run_cbr_adapt.py — Retrieve → Adapt CLI for the CBR Triage System

Orchestrates the full Retrieve → Adapt cycle:
  1. Build a triage case for a target patient (from SQLite).
  2. Retrieve similar cases via the KG-backed retrieval engine.
  3. Adapt the best-matching case using one of three strategies:
       - null       : return retrieved solution unchanged (baseline).
       - rule_based : deterministic clinical-rule adaptation (CPU-only).
       - llm        : MedGemma LLM adaptation (requires GPU node).

Usage (CPU — login or compute node):
    python scripts/run_cbr_adapt.py --id 141765 --strategy rule_based

Usage (GPU — compute node only):
    python scripts/run_cbr_adapt.py --id 141765 --strategy llm \
        --model_path /N/scratch/alikh/models/google--medgemma-27b-it

Usage (with JSON file from a previous retrieval run):
    python scripts/run_cbr_adapt.py --id 141765 --strategy rule_based \
        --retrieval_json output.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup Paths & Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbr.adaptation import (
    llm_adapt,
    null_adapt,
    rule_based_adapt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CBR-Adapt")

# ---------------------------------------------------------------------------
# Colour helpers (ANSI)
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

STRATEGY_CHOICES = ("null", "rule_based", "llm")
DEFAULT_MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-27b-it"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_target_case(patient_id: int) -> dict:
    """Build the triage case from SQLite for the target patient."""
    from src.inference.build_triage_case import build_triage_case

    return build_triage_case(patient_id)


def _retrieve_cases(age: int, problem: str, patient_id: int, top_k: int) -> list:
    """Retrieve similar cases from the KG."""
    from src.cbr.retrieval import retrieve

    return retrieve(
        name=f"Patient_{patient_id}",
        age=age,
        problem=problem,
        top_k=top_k,
    )


def _load_retrieval_json(filepath: str) -> list:
    """Load pre-computed retrieval results from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def _print_adaptation_result(result: dict, strategy: str) -> None:
    """Pretty-print the adaptation result to console."""
    print(f"\n{BOLD}{CYAN}{'═' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  CBR ADAPTATION RESULT  (strategy: {strategy}){RESET}")
    print(f"{BOLD}{CYAN}{'═' * 72}{RESET}\n")

    if strategy == "rule_based":
        priority = result.get("adapted_priority", "?")
        color = RED if priority == "High" else YELLOW if priority == "Medium" else GREEN
        print(f"  {BOLD}Triage Priority:{RESET} {color}{priority}{RESET}")
        print(f"  {BOLD}New Patient Acuity:{RESET}  {result.get('new_acuity', '?')}")
        print(f"  {BOLD}Retrieved Acuity:{RESET}    {result.get('retrieved_acuity', '?')}")
        print(f"  {BOLD}Acuity Delta:{RESET}        {result.get('acuity_delta', '?')}")

        notes = result.get("comparison_notes", [])
        if notes:
            print(f"\n  {BOLD}{YELLOW}Comparison Notes:{RESET}")
            for note in notes:
                print(f"    • {note}")

        adjustments = result.get("recommended_adjustments", [])
        if adjustments:
            print(f"\n  {BOLD}{YELLOW}Recommended Adjustments:{RESET}")
            for adj in adjustments:
                print(f"    ▸ {adj}")

    elif strategy == "llm":
        sol = result.get("adapted_solution")
        if sol and isinstance(sol, dict) and "raw_text" not in sol:
            print(f"  {BOLD}LLM Adapted Solution:{RESET}")
            print(json.dumps(sol, indent=4))
        elif sol and isinstance(sol, dict) and "raw_text" in sol:
            print(f"  {BOLD}LLM Raw Response:{RESET}")
            print(sol["raw_text"][:2000])
        else:
            print(f"  {BOLD}Prompt generated (no inference performed).{RESET}")
            print(f"  Use --run_local on a GPU node to run MedGemma inference.")

    elif strategy == "null":
        print(f"  {BOLD}Null adaptation — retrieved solution returned unchanged.{RESET}")

    print(f"\n  {BOLD}Adapted at:{RESET} {result.get('adapted_at', '?')}")
    print(f"  {BOLD}Strategy:{RESET}   {result.get('strategy', strategy)}")
    print(f"{BOLD}{CYAN}{'═' * 72}{RESET}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run CBR Retrieve → Adapt cycle for ICU triage."
    )
    parser.add_argument(
        "--id", type=int, required=True, help="Target patient unit stay ID"
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="rule_based",
        help="Adaptation strategy (default: rule_based)",
    )
    parser.add_argument(
        "--retrieval_json",
        type=str,
        default=None,
        help="Path to a pre-computed retrieval JSON file (skip live retrieval)",
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=3,
        help="Number of similar cases to retrieve (default: 3)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to MedGemma weights (for --strategy llm)",
    )
    parser.add_argument(
        "--run_local",
        action="store_true",
        default=False,
        help="Actually run LLM inference locally (GPU node only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write adaptation result to a JSON file",
    )

    args = parser.parse_args()

    # ── Step 1: Build target case ──
    logger.info("Building triage case for Patient %d...", args.id)
    try:
        target_case = _build_target_case(args.id)
    except Exception as e:
        logger.error("Failed to build target case: %s", e)
        sys.exit(1)

    age = target_case["demographics"]["age"]
    problem = (
        target_case["demographics"].get("apache_admission_dx")
        or target_case["demographics"].get("admitting_diagnosis")
        or "General"
    )
    try:
        query_age = int(age)
    except (ValueError, TypeError):
        query_age = 85

    logger.info("Target: Age=%d, Problem='%s'", query_age, problem)

    # ── Step 2: Retrieve similar cases ──
    if args.retrieval_json:
        logger.info("Loading retrieval results from %s", args.retrieval_json)
        raw_results = _load_retrieval_json(args.retrieval_json)
        # Use the source_case from the first result as the retrieved solution
        if not raw_results:
            logger.error("Retrieval JSON is empty.")
            sys.exit(1)
        best_match = raw_results[0]
        retrieved_solution = best_match.get("source_case", best_match)
        logger.info(
            "Using pre-computed match: case_id=%s, score=%.4f",
            best_match.get("case_id", "?"),
            best_match.get("score", 0),
        )
    else:
        logger.info("Retrieving similar cases from KG...")
        results = _retrieve_cases(query_age, problem, args.id, args.retrieval_k)
        if not results:
            logger.warning("No similar cases found. Cannot adapt.")
            sys.exit(0)
        best = results[0]
        retrieved_solution = best.source_case
        logger.info(
            "Best match: case_id=%s, score=%.4f",
            best.case_id,
            best.score,
        )

    # ── Step 3: Adapt ──
    logger.info("Adapting with strategy: %s", args.strategy)

    if args.strategy == "null":
        adapted = null_adapt(target_case, retrieved_solution)
        adapted["strategy"] = "null"
        adapted["adapted_at"] = datetime.now(timezone.utc).isoformat()
    elif args.strategy == "rule_based":
        adapted = rule_based_adapt(target_case, retrieved_solution)
    elif args.strategy == "llm":
        adapted = llm_adapt(
            target_case,
            retrieved_solution,
            model_hint=args.model_path,
            run_local=args.run_local,
        )
    else:
        logger.error("Unknown strategy: %s", args.strategy)
        sys.exit(1)

    # ── Step 4: Output ──
    _print_adaptation_result(adapted, args.strategy)

    if args.output:
        # Serialise — strip the full retrieved solution to keep output manageable
        output_data = {k: v for k, v in adapted.items() if k != "original_retrieved_solution"}
        output_data["target_patient_id"] = args.id
        output_data["query_age"] = query_age
        output_data["query_problem"] = problem
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Adaptation result written to %s", args.output)

    logger.info("Done.")


if __name__ == "__main__":
    main()
