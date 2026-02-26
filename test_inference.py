import os
import torch
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

MODEL_PATH = "/N/scratch/alikh/models/google--medgemma-27b-it"
DB_PATH = "/N/scratch/alikh/eicu_demo.sqlite"
PATIENT_ID = 141765

# 1. Setup Logging Directory
model_dirname = os.path.basename(MODEL_PATH)
log_dir = os.path.join("inference_logs", model_dirname)
os.makedirs(log_dir, exist_ok=True)

def get_patient_profile(patient_id):
    conn = sqlite3.connect(DB_PATH)
    
    # Get basic patient info
    patient = pd.read_sql(f"SELECT * FROM patient WHERE patientunitstayid = {patient_id}", conn).iloc[0]
    
    # Get diagnoses
    diagnoses = pd.read_sql(f"SELECT diagnosisname FROM diagnosis WHERE patientunitstayid = {patient_id}", conn)
    
    # Get medications
    meds = pd.read_sql(f"SELECT drugname, dosage FROM medication WHERE patientunitstayid = {patient_id}", conn)
    
    # Get labs (most recent 10)
    labs = pd.read_sql(f"SELECT labname, labresult, labresultrevisedtimestamp FROM lab WHERE patientunitstayid = {patient_id} ORDER BY labresultrevisedtimestamp DESC LIMIT 10", conn)
    
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
{', '.join(diagnoses['diagnosisname'].tolist()) if not diagnoses.empty else 'None recorded'}

MEDICATIONS:
{meds.to_string(index=False) if not meds.empty else 'None recorded'}

RECENT LAB RESULTS:
{labs.to_string(index=False) if not labs.empty else 'None recorded'}
"""
    return profile

def run_inference():
    print(f"Loading data for patient {PATIENT_ID}...")
    profile = get_patient_profile(PATIENT_ID)
    
    print(f"Loading model from {MODEL_PATH}...")
    print("This may take a few minutes on the AMD GPU...")
    
    # Using bfloat16 for memory efficiency on MI210
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    prompt = f"""You are a helpful medical assistant. Below is a clinical profile for a patient. 
Translate this clinical data into a layman's report that the patient's family can understand. 
Explain the condition, current treatments, and labs in simple terms.

{profile}

LAYMAN'S REPORT:"""

    print("Generating report...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"patient_{PATIENT_ID}_{timestamp}.txt")
    
    with open(log_file, "w") as f:
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"MODEL: {MODEL_PATH}\n")
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
