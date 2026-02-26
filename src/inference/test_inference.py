import os
import torch
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-4b-it"
FALLBACK_MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-27b-it"
DB_PATH = "/N/scratch/alikh/eicu_demo.sqlite"
PATIENT_ID = 141765
OFFLOAD_DIR = "/N/scratch/alikh/models/offload"

# 1. Setup Logging Directory
os.makedirs("logs/inference", exist_ok=True)


def ensure_model_dir_ready(model_path: str) -> None:
    # Minimal required files for HF local model loading.
    required = ["config.json"]
    missing = [name for name in required if not os.path.exists(os.path.join(model_path, name))]
    if missing:
        raise FileNotFoundError(
            f"Model directory is incomplete: {model_path}. Missing: {', '.join(missing)}. "
            "Re-download model weights/tokenizer files from Hugging Face."
        )

def get_patient_profile(patient_id):
    conn = sqlite3.connect(DB_PATH)
    
    # Get basic patient info
    patient = pd.read_sql(f"SELECT * FROM patient WHERE patientunitstayid = {patient_id}", conn).iloc[0]
    
    # Get diagnoses
    diagnoses = pd.read_sql(f"SELECT diagnosisstring FROM diagnosis WHERE patientunitstayid = {patient_id}", conn)
    
    # Get medications
    meds = pd.read_sql(f"SELECT drugname, dosage FROM medication WHERE patientunitstayid = {patient_id}", conn)
    
    # Get labs (most recent 10)
    labs = pd.read_sql(f"SELECT labname, labresult, labresultrevisedoffset FROM lab WHERE patientunitstayid = {patient_id} ORDER BY labresultrevisedoffset DESC LIMIT 10", conn)
    
    conn.close()
    
    profile = f"""
PATIENT PROFILE
---------------
ID: {patient['patientunitstayid']}
Age: {patient['age']}
Gender: {patient['gender']}
Ethnicity: {patient['ethnicity']}
Admission Diagnosis: {patient['apacheadmissiondx']}

DIAGNOSES:
{', '.join(diagnoses['diagnosisstring'].tolist()) if not diagnoses.empty else 'None recorded'}

MEDICATIONS:
{meds.to_string(index=False) if not meds.empty else 'None recorded'}

RECENT LAB RESULTS:
{labs.to_string(index=False) if not labs.empty else 'None recorded'}
"""
    return profile

def run_inference():
    print(f"Loading data for patient {PATIENT_ID}...")
    profile = get_patient_profile(PATIENT_ID)
    
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    primary_model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    fallback_model_path = os.getenv("FALLBACK_MODEL_PATH", FALLBACK_MODEL_PATH).strip()
    active_model_path = primary_model_path
    ensure_model_dir_ready(primary_model_path)
    print(f"Loading model from {active_model_path}...")
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {total:.1f} GiB")
    print("This may take a few minutes on the NVIDIA A100...")

    # Clear any CUDA allocator fragmentation before heavy load
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(primary_model_path, use_fast=False)

    try:
        from transformers import BitsAndBytesConfig

        # ---- Two-phase load to avoid HF loader OOM on 40 GB A100 ----
        # The HF loader's materialization always stages tensors on the target
        # device BEFORE quantizing, so any GPU-targeted load OOMs on 27B.
        #
        # Phase 1: load the full model in float16 onto CPU RAM (122 GB available).
        #          No GPU is touched here — safe for any model size.
        print("Phase 1: loading model weights into CPU RAM (float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            primary_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print("Phase 1 complete. Quantizing and moving to GPU...")

        # Phase 2: quantize in-place then dispatch to GPU.
        # bitsandbytes' replace_with_bnb_linear works on a CPU model and converts
        # linear layers to 8-bit, then .cuda() moves the already-small model.
        from bitsandbytes.nn import Linear8bitLt
        from bitsandbytes import replace_with_bnb_linear

        model = replace_with_bnb_linear(
            model,
            modules_to_not_convert=["lm_head"],
            quantization_config=None,
            has_been_replaced=False,
        )[0]
        model = model.cuda()
        model.eval()

        # --- GPU memory diagnostics after load ---
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            resrv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f} GiB allocated, {resrv:.2f} GiB reserved")
    except Exception as exc:
        if not fallback_model_path or fallback_model_path == primary_model_path:
            raise RuntimeError(f"Primary model load failed and no distinct fallback is configured: {exc}") from exc
        print(f"Primary model load failed, trying fallback model {fallback_model_path}: {exc}")
        torch.cuda.empty_cache()
        active_model_path = fallback_model_path
        ensure_model_dir_ready(active_model_path)
        tokenizer = AutoTokenizer.from_pretrained(active_model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            active_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model = model.cuda()
        model.eval()
    
    prompt = f"""You are a helpful medical assistant. Below is a clinical profile for a patient. 
Translate this clinical data into a layman's report that the patient's family can understand. 
Explain the condition, current treatments, and labs in simple terms.

{profile}

LAYMAN'S REPORT:"""

    print("Generating report...")
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    report = response.split("LAYMAN'S REPORT:")[-1].strip()
    
    # 2. Save Interaction
    model_dirname = os.path.basename(active_model_path)
    log_dir = os.path.join("logs/inference", model_dirname)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"patient_{PATIENT_ID}_{timestamp}.txt")
    
    with open(log_file, "w") as f:
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"MODEL: {active_model_path}\n")
        f.write("-" * 50 + "\n")
        f.write("INPUT PROMPT:\n")
        f.write(prompt + "\n")
        f.write("-" * 50 + "\n")
        f.write("MODEL OUTPUT:\n")
        f.write(report + "\n")
        
    print(f"\nSuccess! Log saved to: {log_file}")
    print("\nGenerated Report Snippet:\n")
    print(report[:500] + "...")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_inference()
    else:
        print("ERROR: GPU not detected. This script MUST run on a GPU node using srun.")
