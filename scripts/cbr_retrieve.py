#!/usr/bin/env python3
"""
cbr_retrieve.py — CLI for KG-backed patient case retrieval.

Usage:
    python scripts/cbr_retrieve.py --name "John Doe" --age 72 --problem "sepsis"
    python scripts/cbr_retrieve.py --name "Jane Doe" --age 45 --problem "hypertension" --top_k 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import src.cbr
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.cbr.retrieval import retrieve
except ImportError as e:
    print(f"ERROR: Could not import src.cbr.retrieval: {e}", file=sys.stderr)
    print(
        "Ensure you are running from the project root and venv is active.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve similar patient cases using KG discovery and composite similarity scoring."
    )

    # Mandatory fields
    parser.add_argument("--name", required=True, help="Patient identifier (mandatory)")
    parser.add_argument(
        "--age", type=int, required=True, help="Patient age in years (mandatory)"
    )
    parser.add_argument(
        "--problem", required=True, help="Clinical problem keyword (mandatory)"
    )

    # Optional parameters
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=15,
        help="Age range for KG filtering (default: ±15)",
    )
    parser.add_argument(
        "--gender", type=str, default=None,
        help="Query patient gender (e.g. 'Female', 'Male') — enables gender matching",
    )
    parser.add_argument(
        "--comorbidities", type=str, default=None,
        help="Comma-separated active comorbidity flag names (e.g. 'chf,ckd') — enables Jaccard scoring",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed similarity breakdown",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as raw JSON"
    )
    parser.add_argument(
        "--full", action="store_true", help="Include full case structure in JSON output"
    )

    args = parser.parse_args()

    # Parse comorbidity flags into a set
    comorbidity_flags = None
    if args.comorbidities:
        comorbidity_flags = {f.strip().lower() for f in args.comorbidities.split(",") if f.strip()}

    try:
        # Perform retrieval
        results = retrieve(
            name=args.name,
            age=args.age,
            problem=args.problem,
            top_k=args.top_k,
            age_tolerance=args.tolerance,
            gender=args.gender,
            comorbidity_flags=comorbidity_flags,
        )

        if args.json:
            # Machine readable output
            output = []
            for r in results:
                item = {
                    "case_id": r.case_id,
                    "score": r.score,
                    "age": r.patient_age,
                    "gender": r.patient_gender,
                    "details": r.details,
                }
                if args.full:
                    item["source_case"] = r.source_case
                output.append(item)

            print(json.dumps(output, indent=2))
            return

        # Human readable output
        print(f"\n{'=' * 80}")
        print(f" CBR RETRIEVAL RESULTS")
        print(f" Query: Name={args.name}, Age={args.age}, Problem='{args.problem}'")
        print(f"{'=' * 80}\n")

        if not results:
            print(" No similar cases found in the Knowledge Graph.")
            return

        for i, res in enumerate(results, 1):
            print(f" {i}. [Score: {res.score:.4f}] Case ID: {res.case_id}")
            print(
                f"    Demographics: {res.patient_age or '??'} yrs | {res.patient_gender or 'Unknown'}"
            )
            if args.verbose:
                details = ", ".join([f"{k}={v}" for k, v in res.details.items()])
                print(f"    Breakdown   : {details}")
            print(f"    {'-' * 40}")

        print(f"\n Found {len(results)} matches.")

    except Exception as e:
        print(f"\n ERROR during retrieval: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
