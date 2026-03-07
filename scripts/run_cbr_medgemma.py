#!/usr/bin/env python3
"""
run_cbr_medgemma.py — End-to-End CBR Cycle with MedGemma LLM

This script orchestrates the full CBR process:
1.  **Build**: Extracts triage features for a target patient (from SQLite).
2.  **Retrieve**: Finds similar past cases using the KG-backed retrieval engine.
3.  **Reuse/Revise**: Uses MedGemma (LLM) to analyze the new case in context of the similar cases.

Usage:
    python scripts/run_cbr_medgemma.py --id 141765 --model_path /N/scratch/alikh/models/google--medgemma-27b-it
"""

import argparse
import json
import logging
import os
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# ---------------------------------------------------------------------------
# Setup Paths & Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import internal modules
try:
    from src.inference.build_triage_case import build_triage_case
    from src.cbr.retrieval import retrieve, SimilarityResult
except ImportError as e:
    sys.exit(f"ERROR: Could not import project modules: {e}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-27b-it"
# Fallback if 27b not found, though user can override
FALLBACK_MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-2b-it"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CBR-MedGemma")


# ---------------------------------------------------------------------------
# Helper: Prompt Construction
# ---------------------------------------------------------------------------

def format_case_summary(case_data: Dict[str, Any]) -> str:
    """Creates a concise textual representation of a patient case for the LLM."""
    demos = case_data.get("demographics", {})
    acute = case_data.get("acute_physiology_24h", {})
    labs = case_data.get("key_labs_24h", {})
    context = case_data.get("triage_context", {})
    
    # Extract key vitals
    vitals = []
    if acute.get("heart_rate", {}).get("max"):
        vitals.append(f"HR Max: {acute['heart_rate']['max']}")
    if acute.get("map", {}).get("min"):
        vitals.append(f"MAP Min: {acute['map']['min']}")
    if acute.get("gcs", {}).get("total"):
        vitals.append(f"GCS: {acute['gcs']['total']}")
    
    # Extract Severity Flags
    flags = context.get("severity_flags", [])
    
    # Outcome (only relevant for retrieved cases)
    outcome = ""
    if "icu_los" in case_data:
        los_days = case_data["icu_los"].get("los_days")
        mortality = case_data.get("apache", {}).get("actual_hospital_mortality", "Unknown")
        outcome = f"\nOUTCOME:\n- ICU LOS: {los_days} days\n- Mortality: {mortality}"

    summary = f"""ID: {case_data.get("metadata", {}).get("patient_unit_stay_id")}
- Age/Gender: {demos.get("age")} / {demos.get("gender")}
- Diagnosis: {demos.get("apache_admission_dx", "Unknown")}
- Key Vitals (First 24h): {', '.join(vitals)}
- Severity Flags: {', '.join(flags) if flags else "None"}
- Organ Support: {', '.join([k for k,v in case_data.get("organ_support", {}).items() if (isinstance(v,bool) and v) or (isinstance(v,dict) and v.get("flag"))]) or "None"}{outcome}
"""
    return summary


def build_cbr_prompt(target_case: Dict[str, Any], retrieved_cases: List[SimilarityResult]) -> str:
    """Constructs the prompt including the target case and retrieved examples."""
    
    target_summary = format_case_summary(target_case)
    
    example_text = ""
    for i, res in enumerate(retrieved_cases, 1):
        # We need the full source case which our updated retrieval now provides
        if getattr(res, 'source_case', None):
            case_text = format_case_summary(res.source_case)
            example_text += f"\n--- SIMILAR CASE {i} (Similarity: {res.score:.2f}) ---\n{case_text}"
    
    system_instruction = (
        "You are an expert Critical Care AI assistant. "
        "Your goal is to assess a new ICU patient based on their clinical data and outcome patterns from similar past cases.\n"
        "1. Analyze the New Patient's severity.\n"
        "2. Compare them to the Similar Cases provided.\n"
        "3. Predict potential risks and suggest a triage priority (High/Medium/Low)."
    )

    prompt = f"""<start_of_turn>user
{system_instruction}

=== SIMILAR PAST CASES ===
{example_text}

=== NEW PATIENT ===
{target_summary}

Based on the similar cases and the new patient's data, provide a rapid triage assessment.
<end_of_turn>
<start_of_turn>model
"""
    return prompt


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CBR Cycle with MedGemma")
    parser.add_argument("--id", type=int, required=True, help="Target Patient Unit Stay ID")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to MedGemma weights")
    parser.add_argument("--retrieval_k", type=int, default=3, help="Number of similar cases to retrieve")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available (default: True)")
    
    args = parser.parse_args()

    # 1. BUILD TARGET CASE
    logger.info(f"Building triage case for Patient {args.id}...")
    try:
        target_case = build_triage_case(args.id)
    except Exception as e:
        logger.error(f"Failed to build case: {e}")
        return

    # Extract query parameters
    age = target_case["demographics"]["age"]
    # Prefer apache diagnosis, fallback to free text or generic
    problem = target_case["demographics"].get("apache_admission_dx") or \
              target_case["demographics"].get("admitting_diagnosis") or \
              "General"
    
    # Handle non-numeric age (e.g. "80+")
    try:
        query_age = int(age)
    except ValueError:
        query_age = 85 # meaningful default for elderly

    logger.info(f"Target Case Built. Querying for: Age={query_age}, Problem='{problem}'")

    # 2. RETRIEVE SIMILAR CASES
    logger.info("Retrieving similar cases from KG...")
    # Note: verify that retrieve() is returning the source_case in the objects (we patched this earlier)
    results = retrieve(
        name=f"Patient_{args.id}", 
        age=query_age, 
        problem=problem, 
        top_k=args.retrieval_k
    )
    
    if not results:
        logger.warning("No similar cases found. Inference will proceed without few-shot examples.")
    else:
        logger.info(f"Found {len(results)} similar cases.")

    # 3. CONSTRUCT PROMPT
    prompt_text = build_cbr_prompt(target_case, results)
    
    print("\n" + "="*60)
    print(" GENERATED PROMPT PREVIEW ")
    print("="*60)
    print(prompt_text)
    print("="*60 + "\n")

    # 4. LOAD LLM & INFERENCE
    logger.info(f"Loading MedGemma from {args.model_path}...")
    
    # Check GPU
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Inference Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Starting Generation...")
    print("\n" + "="*20 + " MODEL OUTPUT " + "="*20)
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=512, 
            temperature=0.2, 
            do_sample=True
        )
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
