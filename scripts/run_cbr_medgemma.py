#!/usr/bin/env python3
"""
run_cbr_medgemma.py — End-to-End CBR Cycle with MedGemma LLM

This script orchestrates the full CBR process:
1.  **Build**: Extracts triage features for a target patient (from SQLite).
2.  **Retrieve**: Finds similar past cases using the KG-backed retrieval engine.
3.  **Reuse/Revise**: Uses MedGemma (LLM) to analyze the new case in context of the similar cases.

Usage:
    python scripts/run_cbr_medgemma.py --id 141765 --model_path /N/scratch/alikh/models/google--medgemma-4b
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ---------------------------------------------------------------------------
# Setup Paths & Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import internal modules
try:
    # Removed direct retrieval import, using subprocess to cbr_retrieve.py
    from src.inference import model_state
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
    if acute.get("heart_rate", {}).get("value"):
        vitals.append(f"HR: {acute['heart_rate']['value']}")
    if acute.get("map", {}).get("value"):
        vitals.append(f"MAP: {acute['map']['value']}")
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


def build_cbr_prompt(target_case: Dict[str, Any], retrieved_cases: List[Dict[str, Any]]) -> str:
    """Constructs the prompt including the target case and retrieved examples."""
    
    target_summary = format_case_summary(target_case)
    
    example_text = ""
    for i, res in enumerate(retrieved_cases, 1):
        # res is now a dictionary from the JSON output of cbr_retrieve.py
        if res.get('source_case'):
            case_text = format_case_summary(res['source_case'])
            score = res.get('score', 0.0)
            example_text += f"\n--- SIMILAR CASE {i} (Similarity: {score:.2f}) ---\n{case_text}"
    
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

# ---------------------------------------------------------------------------
# Helper: Build case from CLI args (no DB required)
# ---------------------------------------------------------------------------

def _infer_age_group(age: int) -> str:
    if age < 18:
        return "<18"
    elif age < 40:
        return "18-39"
    elif age < 60:
        return "40-59"
    elif age < 80:
        return "60-79"
    return "80+"


def _build_case_from_args(args) -> Dict[str, Any]:
    """Construct a minimal triage-compatible case dict from CLI args (no DB lookup)."""
    age_int = args.age
    return {
        "metadata": {"patient_unit_stay_id": None, "source": "direct_input"},
        "demographics": {
            "age": age_int,
            "age_raw": args.age_raw or str(age_int),
            "age_group": args.age_group or _infer_age_group(age_int),
            "gender": args.gender or "Unknown",
            "ethnicity": args.ethnicity or "Unknown",
            "apache_admission_dx": args.problem,
            "admitting_diagnosis": args.problem,
            "primary_diagnosis_string": args.problem,
        },
        "triage_context": {"severity_flags": []},
        "acute_physiology_24h": {},
        "organ_support": {},
        "key_labs_24h": {},
        "comorbidities": {"flags": {}, "evidence": {}},
    }


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run CBR Cycle with MedGemma. "
            "Supply a new patient's basic info — the system retrieves similar past cases "
            "from the knowledge graph and adapts a clinical plan using the LLM."
        )
    )
    parser.add_argument("--age", type=int, required=True, help="Patient age in years (e.g. 87)")
    parser.add_argument("--age_raw", type=str, default=None, help="Raw age string if non-numeric (e.g. '> 89')")
    parser.add_argument("--age_group", type=str, default=None, help="Age group label (e.g. '80+'). Auto-inferred if omitted.")
    parser.add_argument("--gender", type=str, default=None, help="Patient gender (e.g. 'Female', 'Male')")
    parser.add_argument("--ethnicity", type=str, default=None, help="Patient ethnicity (e.g. 'Caucasian')")
    parser.add_argument("--problem", type=str, required=True, help="Primary clinical problem / admission diagnosis (e.g. 'Sepsis')")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to MedGemma weights")
    parser.add_argument("--retrieval_k", type=int, default=10, help="Initial number of similar cases to retrieve from KG")
    parser.add_argument("--adapt_k", type=int, default=4, help="Number of refined cases to pass to MedGemma")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available (default: True)")

    args = parser.parse_args()

    # 1. BUILD TARGET CASE from supplied patient info (no database lookup)
    logger.info(f"Building case from direct input: age={args.age}, gender={args.gender}, problem='{args.problem}'")
    target_case = _build_case_from_args(args)

    # Extract query parameters
    problem = args.problem
    query_age = args.age

    logger.info(f"Target Case Built. Querying for: Age={query_age}, Problem='{problem}'")

    # 2. RETRIEVE SIMILAR CASES
    logger.info(f"Retrieving initial cohort of {args.retrieval_k} similar cases from KG using cbr_retrieve.py...")

    # Extract gender and active comorbidity flags from the built target case
    query_gender = target_case.get("demographics", {}).get("gender")
    comorb_flags_dict = target_case.get("comorbidities", {}).get("flags", {})
    active_comorbs = [k for k, v in comorb_flags_dict.items() if v]

    # Call cbr_retrieve.py via subprocess to get JSON output
    retrieve_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "cbr_retrieve.py"),
        "--name", f"Patient_{args.age}_{args.problem.replace(' ', '_')}",
        "--age", str(query_age),
        "--problem", problem,
        "--top_k", str(args.retrieval_k),
        "--json",
        "--full",
    ]
    if query_gender and query_gender != "Unknown":
        retrieve_cmd += ["--gender", query_gender]
    if active_comorbs:
        retrieve_cmd += ["--comorbidities", ",".join(active_comorbs)]
    
    try:
        retrieval_proc = subprocess.run(
            retrieve_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        results = json.loads(retrieval_proc.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Retrieval script failed: {e.stderr}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse retrieval JSON: {e}\nOutput was: {retrieval_proc.stdout}")
        return
    
    if not results:
        logger.warning("No similar cases found. Inference will proceed without few-shot examples.")
    else:
        logger.info(f"Found {len(results)} similar cases. Filtering down to top {args.adapt_k} for MedGemma...")
        # Narrow down initial cohort to top target_k cases
        results = results[:args.adapt_k]

    # 3. CONSTRUCT PROMPT
    prompt_text = build_cbr_prompt(target_case, results)
    
    print("\n" + "="*60)
    print(" GENERATED PROMPT PREVIEW ")
    print("="*60)
    print(prompt_text)
    print("="*60 + "\n")

    # 5. LOAD LLM & INFERENCE
    logger.info(f"Loading MedGemma from {args.model_path}...")
    
    # Check GPU
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Inference Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto" if device == "cuda" else None,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Starting Generation...")
    print("\n" + "="*20 + " MODEL OUTPUT " + "="*20)
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    # Provide the pad_token_id to the model generate call
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Fallback dtype for generation
    # if using float16 we need to cast inputs carefully or avoid multinomial errors
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer, 
            max_new_tokens=512, 
            temperature=0.2, 
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
