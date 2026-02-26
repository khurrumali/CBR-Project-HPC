#!/usr/bin/env python3
"""
test_kg.py  –  Verification tests for the eICU Oxigraph knowledge graph.

Usage:
    python test_kg.py
"""

import os, sys, sqlite3
from dotenv import load_dotenv
from pyoxigraph import Store

from rdf_schema import SPARQL_PREFIXES

# Add OWL prefix that SPARQL_PREFIXES doesn't include
SPARQL_HEADER = SPARQL_PREFIXES + "\nPREFIX owl:  <http://www.w3.org/2002/07/owl#>"

load_dotenv()
STORE_PATH = os.getenv("OXIGRAPH_STORE_PATH", "/N/scratch/alikh/oxigraph_store")
DB_URL     = os.getenv("DATABASE_URL", "")
DB_PATH    = DB_URL.replace("sqlite:///", "", 1) if DB_URL.startswith("sqlite:///") else "/N/scratch/alikh/eicu_demo.sqlite"

# ── Helpers ─────────────────────────────────────────────────────────────────

passed = 0
failed = 0

def _q(store, sparql):
    if "PREFIX" not in sparql.upper():
        sparql = SPARQL_HEADER + "\n" + sparql
    return list(store.query(sparql))

def _count(store, sparql):
    rows = _q(store, sparql)
    return int(rows[0]["c"].value) if rows else 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓  {name}")
        passed += 1
    else:
        print(f"  ✗  {name}  — {detail}")
        failed += 1


# ── Tests ───────────────────────────────────────────────────────────────────

def main():
    global passed, failed

    if not os.path.isdir(STORE_PATH):
        print(f"ERROR: Store not found at {STORE_PATH}. Run build_kg.py first.")
        sys.exit(1)

    store = Store(STORE_PATH)
    total = len(store)
    print(f"\n  Opened store: {total:,} triples\n")

    # 1. Total triple count
    check("Total triples > 30,000", total > 30_000, f"got {total}")

    # 2. Hospital count = 186
    n = _count(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s rdf:type eicu:Hospital }")
    check("Hospital count = 186", n == 186, f"got {n}")

    # 3. Patient count = 2520
    n = _count(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s rdf:type eicu:Patient }")
    check("Patient count = 2520", n == 2520, f"got {n}")

    # 4. Diagnosis count > 500 (de-duplicated by diagnosisstring+icd9)
    n = _count(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s rdf:type eicu:Diagnosis }")
    check(f"Diagnosis count > 500 (got {n:,})", n > 500)

    # 5. OrganSystem count > 5
    n = _count(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s rdf:type eicu:OrganSystem }")
    check(f"OrganSystem count > 5 (got {n})", n > 5)

    # 6. RawDrugName count > 100
    n = _count(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s rdf:type eicu:RawDrugName }")
    check(f"RawDrugName count > 100 (got {n:,})", n > 100)

    # 7. SKOS Concept count > 100
    n = _count(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s rdf:type skos:Concept }")
    check(f"SKOS Concept count > 100 (got {n:,})", n > 100)

    # 8. Ontology classes defined (ASK returns QueryBoolean, not iterable)
    for cls in ["eicu:Hospital", "eicu:Patient", "eicu:Diagnosis", "eicu:OrganSystem", "eicu:RawDrugName", "skos:Concept"]:
        sparql = f"{SPARQL_HEADER}\nASK {{ {cls} rdf:type owl:Class }}"
        try:
            result = bool(store.query(sparql))
        except Exception:
            result = False
        check(f"OWL class defined: {cls}", result)

    # 9. Every diagnosis has a belongsTo edge
    total_diag = _count(store, "SELECT (COUNT(DISTINCT ?d) AS ?c) WHERE { ?d rdf:type eicu:Diagnosis }")
    with_bt    = _count(store, "SELECT (COUNT(DISTINCT ?d) AS ?c) WHERE { ?d rdf:type eicu:Diagnosis ; eicu:belongsTo ?os }")
    check(f"All diagnoses have belongsTo ({with_bt:,}/{total_diag:,})", with_bt == total_diag)

    # 10. SKOS lookup: hypertension returns results
    rows = _q(store, """
        SELECT DISTINCT ?prefLabel WHERE {
            ?c rdf:type skos:Concept ;
               skos:altLabel ?alt ;
               skos:prefLabel ?prefLabel .
            FILTER(CONTAINS(LCASE(?alt), "hypertension"))
        }
    """)
    check(f"SKOS lookup 'hypertension' returns results ({len(rows)} concepts)", len(rows) > 0)

    # 11. Patients found via SKOS hypertension lookup
    rows = _q(store, """
        SELECT (COUNT(DISTINCT ?pat) AS ?c) WHERE {
            ?concept skos:altLabel ?alt ; skos:prefLabel ?pl .
            FILTER(CONTAINS(LCASE(?alt), "hypertension"))
            ?diag eicu:problem ?pl .
            ?pat eicu:hasDiagnosis ?diag .
        }
    """)
    n = int(rows[0]["c"].value) if rows else 0
    check(f"Patients with hypertension via SKOS ({n:,})", n > 0)

    # 12. Round-trip: patient 141765 data matches SQLite
    if os.path.isfile(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT gender, age, ethnicity, hospitalid FROM patient WHERE patientunitstayid=141765")
        row = cur.fetchone()
        conn.close()
        if row:
            gender_sql, age_sql, eth_sql, hid_sql = row
            # Check gender (use full URI since prefixed names can't contain '/')
            rdf_rows = _q(store, "SELECT ?v WHERE { <http://eicu.mit.edu/data/patient/141765> eicu:gender ?v }")
            rdf_gender = rdf_rows[0]["v"].value if rdf_rows else None
            check(f"Patient 141765 gender matches (SQL={gender_sql}, RDF={rdf_gender})", rdf_gender == gender_sql)

            # Check admittedTo
            rdf_rows = _q(store, "SELECT ?h WHERE { <http://eicu.mit.edu/data/patient/141765> eicu:admittedTo ?h }")
            if rdf_rows:
                rdf_hid = rdf_rows[0]["h"].value.split("/")[-1]
                check(f"Patient 141765 hospital matches (SQL={hid_sql}, RDF={rdf_hid})", str(hid_sql) == rdf_hid)
            else:
                check("Patient 141765 hospital match", False, "no admittedTo edge")
        else:
            check("Patient 141765 exists in SQLite", False, "not found")
    else:
        print("  ⊘  Skipping round-trip test (SQLite DB not accessible)")

    # ── Summary ──
    print(f"\n{'═'*55}")
    print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'═'*55}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
