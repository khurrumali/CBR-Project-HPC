#!/usr/bin/env python3
"""
utils.py — Shared helper functions for eICU case construction.

Used by both build_case.py (general case) and build_triage_case.py (triage features).
"""

import os
import re
import sqlite3

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Database Configuration
# ──────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH = (
    DATABASE_URL.replace("sqlite:///", "")
    if DATABASE_URL.startswith("sqlite:///")
    else DATABASE_URL
)

DATASET_TAG = "eicu-crd-demo-2.0.1"

# ──────────────────────────────────────────────
# Chronic-condition keywords (lowercase)
# ──────────────────────────────────────────────
CHRONIC_KEYWORDS = [
    "chronic", "diabetes", "copd", "hypertension", "renal failure",
    "heart failure", "cirrhosis", "asthma", "hiv", "hepatitis",
    "immunosuppression", "dialysis", "transplant", "cancer", "malignancy",
    "coronary artery disease", "atrial fibrillation", "hypothyroid",
    "hyperthyroid", "seizure", "epilepsy", "dementia", "alzheimer",
    "parkinson", "rheumatoid", "lupus", "scleroderma",
]

# ──────────────────────────────────────────────
# Drug-category mapping (patterns → category)
# ──────────────────────────────────────────────
DRUG_CATEGORY_PATTERNS = [
    (r"warfarin|heparin|enoxaparin|rivaroxaban|apixaban|coumadin", "anticoagulant"),
    (r"aspirin|clopidogrel|plavix|ticagrelor", "antiplatelet"),
    (r"metoprolol|atenolol|propranolol|carvedilol|bisoprolol", "beta_blocker"),
    (r"diltiazem|verapamil|amlodipine|nifedipine", "calcium_channel_blocker"),
    (r"lisinopril|enalapril|ramipril|captopril|losartan|valsartan", "ace_arb"),
    (r"furosemide|bumetanide|hydrochlorothiazide|spironolactone|lasix", "diuretic"),
    (r"insulin|metformin|glipizide|glyburide", "antidiabetic"),
    (r"amiodarone|lidocaine|digoxin|adenosine", "antiarrhythmic"),
    (r"morphine|fentanyl|hydromorphone|oxycodone|acetaminophen|ibuprofen|ketorolac", "analgesic"),
    (r"vancomycin|ceftriaxone|meropenem|piperacillin|azithromycin|levofloxacin|ciprofloxacin|metronidazole", "antibiotic"),
    (r"omeprazole|pantoprazole|famotidine|ranitidine|lansoprazole", "gi_protective"),
    (r"propofol|midazolam|lorazepam|diazepam|dexmedetomidine", "sedative"),
    (r"norepinephrine|epinephrine|vasopressin|dopamine|dobutamine|phenylephrine", "vasopressor_inotrope"),
    (r"albuterol|ipratropium|budesonide|fluticasone|montelukast", "pulmonary"),
    (r"prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone", "corticosteroid"),
    (r"simvastatin|atorvastatin|rosuvastatin|pravastatin", "statin"),
]

# Vasopressor-specific regex (shared between build_case and build_triage_case)
VASOPRESSOR_REGEX = re.compile(
    r"norepinephrine|epinephrine|vasopressin|dopamine|dobutamine|"
    r"phenylephrine|milrinone|levophed|neosynephrine",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────
# carePlanGeneral mappings
# ──────────────────────────────────────────────
EXAM_GROUP_MAP = {
    "Airway": "airway",
    "Ventilation": "ventilation",
    "Neurological": "neuro",
    "Cardiac": "cardiac",
    "GI / Nutrition": "gi_nutrition",
    "Renal": "renal",
    "Infectious Disease": "infectious_disease",
    "Skin / Wound": "skin_wound",
    "Activity": "activity",
    "Safety": "safety",
}

GOAL_GROUPS = {
    "Care Limitation", "Code Status", "Advance Directive",
    "Goals of Care", "Discharge Planning", "Prognosis",
}

# ──────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────

def safe_int(val):
    """Parse to int, return None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val):
    """Parse to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_age(val):
    """Handle eICU age strings: '87' → 87, '> 89' → 90, '' → None."""
    if val is None:
        return None
    val = str(val).strip()
    if not val:
        return None
    if ">" in val:
        return 90  # eICU convention for ages > 89
    return safe_int(val)


def minutes_to_hours(minutes):
    """Convert offset in minutes to hours, rounded to 1 decimal."""
    if minutes is None:
        return None
    try:
        return round(float(minutes) / 60.0, 1)
    except (ValueError, TypeError):
        return None


def compute_bmi(height_cm, weight_kg):
    """BMI = weight(kg) / height(m)^2. Returns rounded float or None."""
    h = safe_float(height_cm)
    w = safe_float(weight_kg)
    if h and w and h > 0 and w > 0:
        h_m = h / 100.0
        bmi = w / (h_m * h_m)
        return round(bmi, 1)
    return None


def is_chronic_keyword(text):
    """Check if text contains chronic-condition markers."""
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in CHRONIC_KEYWORDS)


def classify_drug_category(drug_name):
    """Map a drug name to a therapeutic category, or 'other'."""
    if not drug_name:
        return "other"
    lower = drug_name.lower()
    for pattern, category in DRUG_CATEGORY_PATTERNS:
        if re.search(pattern, lower):
            return category
    return "other"


def age_group(age):
    """Bucket age into comparison groups."""
    if age is None:
        return "unknown"
    if age < 40:
        return "<40"
    if age < 65:
        return "40-64"
    if age < 80:
        return "65-79"
    return "80+"


# ──────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────

def get_connection():
    """Return a sqlite3 connection to the eICU database."""
    if not DB_PATH or not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at '{DB_PATH}'. Check DATABASE_URL in .env"
        )
    return sqlite3.connect(DB_PATH)


def fetch_rows(conn, table, patient_id):
    """SELECT * FROM table WHERE patientunitstayid = ?  →  list of dicts."""
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT * FROM [{table}] WHERE patientunitstayid = ?", (patient_id,)
    )
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def fetch_rows_24h(conn, table, patient_id, offset_col, window_min=0, window_max=1440):
    """SELECT * FROM table WHERE patientunitstayid = ? AND offset BETWEEN 0 AND 1440."""
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT * FROM [{table}] WHERE patientunitstayid = ? "
        f"AND [{offset_col}] BETWEEN ? AND ?",
        (patient_id, window_min, window_max),
    )
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def fetch_patient(conn, patient_id):
    """Fetch single patient row or None."""
    rows = fetch_rows(conn, "patient", patient_id)
    return rows[0] if rows else None


def table_exists(conn, table):
    """Check if a table exists in the SQLite database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=? COLLATE NOCASE",
        (table,),
    )
    return cur.fetchone() is not None
