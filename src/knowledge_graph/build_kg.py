#!/usr/bin/env python3
"""
build_kg.py  –  ETL pipeline: eICU SQLite  ->  Oxigraph RDF Knowledge Graph.

Usage:
    python build_kg.py              # full build (creates / replaces store)
    python build_kg.py --stats      # print stats from existing store only

Reads: DATABASE_URL, OXIGRAPH_STORE_PATH from .env
"""

import os, sys, time, sqlite3, re, hashlib
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from pyoxigraph import Store, Quad, DefaultGraph

from rdf_schema import (
    # Namespace strings
    EICU, DATA, SKOS,
    # Class nodes
    CLASS_HOSPITAL, CLASS_PATIENT, CLASS_DIAGNOSIS,
    CLASS_ORGAN_SYSTEM, CLASS_RAW_DRUG, SKOS_CONCEPT,
    # Data-property predicates
    EICU_BED_CATEGORY, EICU_TEACHING_STATUS, EICU_REGION,
    EICU_AGE, EICU_GENDER, EICU_ETHNICITY, EICU_ICU_MORTALITY, EICU_APACHE_SCORE,
    EICU_DIAGNOSIS_STRING, EICU_ICD9_CODE, EICU_ORGAN_SYSTEM_NAME,
    EICU_CATEGORY, EICU_PROBLEM, EICU_DETAIL, EICU_QUALIFIER,
    EICU_DIAGNOSIS_PRIORITY,
    SKOS_PREF_LABEL, SKOS_ALT_LABEL, SKOS_BROADER, SKOS_EXACT_MATCH,
    # Object-property predicates
    EICU_ORDERED, EICU_ADMITTED_TO, EICU_CONFIRMED_INFUSION,
    EICU_HAS_DIAGNOSIS, EICU_BELONGS_TO,
    # Helpers
    RDF_TYPE, RDFS_LABEL,
    hospital_uri, patient_uri, diagnosis_uri,
    organ_system_uri, drug_uri, concept_uri, category_uri, detail_uri, icd9_uri,
    xsd_string, xsd_integer, xsd_float, xsd_boolean,
    insert_ontology_triples, SPARQL_PREFIXES,
)


# ── Utilities ───────────────────────────────────────────────────────────────

def _parse_age(raw: str | None) -> int | None:
    """Convert eICU age text (e.g. '> 89', '67', '') to int."""
    if not raw or raw.strip() == "":
        return None
    raw = raw.strip()
    if raw.startswith(">"):
        return 90          # convention: "> 89" -> 90
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_diag_parts(diagnosisstring: str) -> dict:
    """Split pipe-delimited diagnosis string into named parts."""
    parts = [p.strip() for p in diagnosisstring.split("|")]
    return {
        "organ_system": parts[0] if len(parts) > 0 else None,
        "category":     parts[1] if len(parts) > 1 else None,
        "problem":      parts[2] if len(parts) > 2 else None,
        "detail":       parts[3] if len(parts) > 3 else None,
        "qualifier":    parts[4] if len(parts) > 4 else None,
    }


_DG = DefaultGraph()

def _bulk_add(store: Store, quads: list[Quad]) -> int:
    """Add a list of quads to the store. Returns count added."""
    for q in quads:
        store.add(q)
    return len(quads)


def _t(s, p, o) -> Quad:
    return Quad(s, p, o, _DG)


# ── Phase loaders ───────────────────────────────────────────────────────────

def load_hospitals(store: Store, conn: sqlite3.Connection) -> int:
    """Phase 1: hospitals -> eicu:Hospital nodes."""
    cur = conn.cursor()
    cur.execute("SELECT hospitalid, numbedscategory, teachingstatus, region FROM hospital")
    triples = []
    for hid, beds, teach, region in cur:
        uri = hospital_uri(hid)
        triples.append(_t(uri, RDF_TYPE, CLASS_HOSPITAL))
        if beds:
            triples.append(_t(uri, EICU_BED_CATEGORY, xsd_string(str(beds))))
        if teach:
            status = "Teaching" if teach.lower().startswith("t") else "Non-Teaching"
            triples.append(_t(uri, EICU_TEACHING_STATUS, xsd_string(status)))
        if region:
            triples.append(_t(uri, EICU_REGION, xsd_string(region)))
    return _bulk_add(store, triples)


def load_patients(store: Store, conn: sqlite3.Connection) -> int:
    """Phase 2: patients -> eicu:Patient nodes + admittedTo edges."""
    cur = conn.cursor()
    cur.execute("""
        SELECT p.patientunitstayid, p.age, p.gender, p.ethnicity,
               p.hospitalid, p.unitdischargestatus,
               a.apachescore
        FROM patient p
        LEFT JOIN apachePatientResult a
            ON p.patientunitstayid = a.patientunitstayid
    """)
    triples = []
    for stayid, age_raw, gender, eth, hid, discharge_status, apache in cur:
        uri = patient_uri(stayid)
        triples.append(_t(uri, RDF_TYPE, CLASS_PATIENT))

        age = _parse_age(age_raw)
        if age is not None:
            triples.append(_t(uri, EICU_AGE, xsd_integer(age)))
        if gender:
            triples.append(_t(uri, EICU_GENDER, xsd_string(gender)))
        if eth:
            triples.append(_t(uri, EICU_ETHNICITY, xsd_string(eth)))

        # ICU mortality: True if discharged with status "Expired"
        if discharge_status:
            died = discharge_status.strip().lower() == "expired"
            triples.append(_t(uri, EICU_ICU_MORTALITY, xsd_boolean(died)))

        # APACHE score (may be NULL if no result row)
        if apache is not None:
            triples.append(_t(uri, EICU_APACHE_SCORE, xsd_float(float(apache))))

        # Relationship: admittedTo hospital
        if hid is not None:
            triples.append(_t(uri, EICU_ADMITTED_TO, hospital_uri(hid)))

    return _bulk_add(store, triples)


    Returns (triple_count, organ_system_names, diagnosis_set).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT diagnosisid, patientunitstayid, diagnosisstring,
               icd9code, diagnosispriority
        FROM diagnosis
    """)
    triples = []
    organ_systems: set[str] = set()
    # diagnosis_set: set of (diag_string, icd9)
    diagnosis_set: set[tuple[str, str | None]] = set()

    for did, stayid, dstr, icd9, priority in cur:
        if not dstr:
            continue
        parts = _parse_diag_parts(dstr)
        d_uri = diagnosis_uri(dstr, icd9)

        triples.append(_t(d_uri, RDF_TYPE, CLASS_DIAGNOSIS))
        triples.append(_t(d_uri, EICU_DIAGNOSIS_STRING, xsd_string(dstr)))

        if icd9:
            triples.append(_t(d_uri, EICU_ICD9_CODE, xsd_string(icd9)))
        if parts["organ_system"]:
            triples.append(_t(d_uri, EICU_ORGAN_SYSTEM_NAME, xsd_string(parts["organ_system"])))
            organ_systems.add(parts["organ_system"])
            # belongsTo edge (legacy)
            triples.append(_t(d_uri, EICU_BELONGS_TO, organ_system_uri(parts["organ_system"])))
        
        # ... rest of literal properties ...
        if parts["category"]: triples.append(_t(d_uri, EICU_CATEGORY, xsd_string(parts["category"])))
        if parts["problem"]:  triples.append(_t(d_uri, EICU_PROBLEM, xsd_string(parts["problem"])))
        if parts["detail"]:   triples.append(_t(d_uri, EICU_DETAIL, xsd_string(parts["detail"])))
        if parts["qualifier"]:triples.append(_t(d_uri, EICU_QUALIFIER, xsd_string(parts["qualifier"])))

        # hasDiagnosis edge
        triples.append(_t(patient_uri(stayid), EICU_HAS_DIAGNOSIS, d_uri))

        # diagnosisPriority
        if priority:
            triples.append(_t(d_uri, EICU_DIAGNOSIS_PRIORITY, xsd_string(priority)))

        # Collect for SKOS phase
        diagnosis_set.add((dstr, icd9))

    count = _bulk_add(store, triples)
    return count, organ_systems, diagnosis_set


def load_organ_systems(store: Store, organ_systems: set[str]) -> int:
    """Phase 4: organ systems -> eicu:OrganSystem nodes."""
    triples = []
    for name in sorted(organ_systems):
        uri = organ_system_uri(name)
        triples.append(_t(uri, RDF_TYPE, CLASS_ORGAN_SYSTEM))
        triples.append(_t(uri, RDF_TYPE, SKOS_CONCEPT)) # SKOS!
        triples.append(_t(uri, RDFS_LABEL, xsd_string(name)))
        triples.append(_t(uri, SKOS_PREF_LABEL, xsd_string(name)))
    return _bulk_add(store, triples)


def load_drugs(store: Store, conn: sqlite3.Connection) -> int:
    """Phase 5: medication + infusiondrug -> eicu:RawDrugName nodes + edges."""
    triples = []
    seen_drugs: set[str] = set()

    # --- Medications (ordered drugs) ---
    cur = conn.cursor()
    cur.execute("SELECT patientunitstayid, drugname FROM medication WHERE drugname IS NOT NULL")
    # Also track hospital->drug via patient->hospital
    cur2 = conn.cursor()
    cur2.execute("SELECT patientunitstayid, hospitalid FROM patient")
    patient_hospital = {r[0]: r[1] for r in cur2}

    hospital_drugs: dict[int, set] = defaultdict(set)

    for stayid, dname in cur:
        dname = dname.strip()
        if not dname:
            continue
        d_uri = drug_uri(dname)
        if dname not in seen_drugs:
            triples.append(_t(d_uri, RDF_TYPE, CLASS_RAW_DRUG))
            triples.append(_t(d_uri, RDFS_LABEL, xsd_string(dname)))
            seen_drugs.add(dname)
        # hospital ordered edge
        hid = patient_hospital.get(stayid)
        if hid is not None:
            hospital_drugs[hid].add(dname)

    # Add hospital->drug ordered edges
    for hid, drugs in hospital_drugs.items():
        h_uri = hospital_uri(hid)
        for dname in drugs:
            triples.append(_t(h_uri, EICU_ORDERED, drug_uri(dname)))

    # --- Infusion drugs  -> confirmedInfusion edges ---
    cur.execute("SELECT patientunitstayid, drugname FROM infusiondrug WHERE drugname IS NOT NULL")
    for stayid, dname in cur:
        dname = dname.strip()
        if not dname:
            continue
        d_uri = drug_uri(dname)
        if dname not in seen_drugs:
            triples.append(_t(d_uri, RDF_TYPE, CLASS_RAW_DRUG))
            triples.append(_t(d_uri, RDFS_LABEL, xsd_string(dname)))
            seen_drugs.add(dname)
        triples.append(_t(patient_uri(stayid), EICU_CONFIRMED_INFUSION, d_uri))

    return _bulk_add(store, triples)


def load_skos_concepts(store: Store, diagnosis_set: set[tuple[str, str | None]]) -> int:
    """Phase 6: Hierarchical SKOS clinical concept layer.

    Builds hierarchy: Organ System > Category > Problem > Detail (with optional Qualifier).
    """
    triples = []
    seen = set() # track URIs already defined

    for dstr, icd9 in diagnosis_set:
        parts = _parse_diag_parts(dstr)
        
        # 1. Organ System (level already defined in Phase 4, but we link to it)
        os_name = parts["organ_system"]
        if not os_name: continue
        os_uri = organ_system_uri(os_name)

        # 2. Category
        cat_name = parts["category"]
        if cat_name:
            cat_uri = category_uri(cat_name)
            if cat_uri not in seen:
                triples.append(_t(cat_uri, RDF_TYPE, SKOS_CONCEPT))
                triples.append(_t(cat_uri, SKOS_PREF_LABEL, xsd_string(cat_name)))
                triples.append(_t(cat_uri, SKOS_BROADER, os_uri))
                seen.add(cat_uri)
            parent_uri = cat_uri
        else:
            parent_uri = os_uri

        # 3. Problem
        prob_name = parts["problem"]
        if prob_name:
            prob_uri = concept_uri(prob_name)
            if prob_uri not in seen:
                triples.append(_t(prob_uri, RDF_TYPE, SKOS_CONCEPT))
                triples.append(_t(prob_uri, SKOS_PREF_LABEL, xsd_string(prob_name)))
                triples.append(_t(prob_uri, SKOS_BROADER, parent_uri))
                if icd9:
                    triples.append(_t(prob_uri, SKOS_EXACT_MATCH, icd9_uri(icd9)))
                seen.add(prob_uri)
            parent_uri = prob_uri

        # 4. Detail / Qualifier
        # We combine Detail + Qualifier for the leaf concept if present
        leaf_parts = []
        if parts["detail"]: leaf_parts.append(parts["detail"])
        if parts["qualifier"]: leaf_parts.append(parts["qualifier"])
        
        if leaf_parts:
            leaf_label = " ".join(leaf_parts)
            # URI depends on the full string for leaf uniqueness
            leaf_uri = detail_uri(dstr) 
            if leaf_uri not in seen:
                triples.append(_t(leaf_uri, RDF_TYPE, SKOS_CONCEPT))
                triples.append(_t(leaf_uri, SKOS_PREF_LABEL, xsd_string(leaf_label)))
                triples.append(_t(leaf_uri, SKOS_BROADER, parent_uri))
                triples.append(_t(leaf_uri, EICU_DIAGNOSIS_STRING, xsd_string(dstr)))
                seen.add(leaf_uri)

    return _bulk_add(store, triples)


# ── Store statistics helper ─────────────────────────────────────────────────

def print_stats(store: Store):
    """Print summary statistics from the RDF store."""
    total = len(store)
    print(f"\n{'═'*55}")
    print(f"  RDF Knowledge Graph Statistics")
    print(f"{'═'*55}")
    print(f"  Total triples          : {total:,}")

    classes = [
        ("Hospital",        CLASS_HOSPITAL),
        ("Patient",         CLASS_PATIENT),
        ("Diagnosis",       CLASS_DIAGNOSIS),
        ("OrganSystem",     CLASS_ORGAN_SYSTEM),
        ("RawDrugName",     CLASS_RAW_DRUG),
        ("SKOS Concept",    SKOS_CONCEPT),
    ]
    for label, cls in classes:
        q = f"{SPARQL_PREFIXES}\nSELECT (COUNT(?s) AS ?c) WHERE {{ ?s rdf:type <{cls.value}> }}"
        for row in store.query(q):
            print(f"  {label:20s} : {row['c'].value}")
    print(f"{'═'*55}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    store_path = os.getenv("OXIGRAPH_STORE_PATH", "/N/scratch/alikh/oxigraph_store")
    db_url     = os.getenv("DATABASE_URL", "")

    # Extract SQLite path from DATABASE_URL (sqlite:////path)
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "", 1)
    else:
        db_path = "/N/scratch/alikh/eicu_demo.sqlite"

    if not os.path.isfile(db_path):
        print(f"ERROR: SQLite database not found: {db_path}")
        sys.exit(1)

    stats_only = "--stats" in sys.argv

    # Create / open persistent Oxigraph store
    Path(store_path).mkdir(parents=True, exist_ok=True)

    if stats_only:
        store = Store(store_path)
        print_stats(store)
        return

    # Full build — wipe and recreate
    import shutil
    if Path(store_path).exists():
        shutil.rmtree(store_path)
        Path(store_path).mkdir(parents=True, exist_ok=True)

    store = Store(store_path)
    conn  = sqlite3.connect(db_path)
    t0    = time.time()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  eICU -> Oxigraph RDF Knowledge Graph Builder        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  DB path   : {db_path}")
    print(f"║  Store path: {store_path}")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Phase 0: Ontology definitions
    t = insert_ontology_triples(store)
    print(f"  [0] Ontology definitions       : {t:>8,} triples")

    # Phase 1: Hospitals
    t = load_hospitals(store, conn)
    print(f"  [1] Hospitals                  : {t:>8,} triples")

    # Phase 2: Patients
    t = load_patients(store, conn)
    print(f"  [2] Patients                   : {t:>8,} triples")

    # Phase 3: Diagnoses (also collects organ systems + diagnosis set)
    t, organ_systems, diagnosis_set = load_diagnoses(store, conn)
    print(f"  [3] Diagnoses                  : {t:>8,} triples")

    # Phase 4: Organ Systems
    t = load_organ_systems(store, organ_systems)
    print(f"  [4] Organ Systems              : {t:>8,} triples")

    # Phase 5: Drugs (medications + infusions)
    t = load_drugs(store, conn)
    print(f"  [5] Drugs (meds + infusions)   : {t:>8,} triples")

    # Phase 6: Hierarchical SKOS Concepts
    t = load_skos_concepts(store, diagnosis_set)
    print(f"  [6] SKOS Clinical Concepts     : {t:>8,} triples")

    elapsed = time.time() - t0
    conn.close()

    print(f"\n  ✓ Build completed in {elapsed:.1f}s")
    print_stats(store)

    # Flush
    store.flush()
    print("  ✓ Store flushed to disk.\n")


if __name__ == "__main__":
    main()
