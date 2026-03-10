#!/usr/bin/env python3
"""
query_kg.py  -  SPARQL query interface to the eICU Oxigraph knowledge graph.

Usage:
    python query_kg.py                          # run all demo queries
    python query_kg.py --query "SELECT ..."     # execute one-off SPARQL
    python query_kg.py --interactive             # REPL mode
"""

import os, sys, argparse
from pathlib import Path

# Ensure this file's directory is on sys.path so rdf_schema resolves
# regardless of the working directory from which the script is invoked.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
from pyoxigraph import Store

from rdf_schema import SPARQL_PREFIXES


# == Demo queries ============================================================

DEMO_QUERIES = {
    "1. Hospital count": f"""
{SPARQL_PREFIXES}
SELECT (COUNT(DISTINCT ?h) AS ?hospitals) WHERE {{
    ?h rdf:type eicu:Hospital .
}}
""",

    "2. Patient count": f"""
{SPARQL_PREFIXES}
SELECT (COUNT(DISTINCT ?p) AS ?patients) WHERE {{
    ?p rdf:type eicu:Patient .
}}
""",

    "3. Organ system breakdown (top 10)": f"""
{SPARQL_PREFIXES}
SELECT ?organ (COUNT(?d) AS ?count)
WHERE {{
    ?d rdf:type eicu:Diagnosis ;
       eicu:belongsTo ?os .
    ?os rdfs:label ?organ .
}}
GROUP BY ?organ
ORDER BY DESC(?count)
LIMIT 10
""",

    "4. SKOS lookup: hypertension": f"""
{SPARQL_PREFIXES}
SELECT DISTINCT ?concept ?prefLabel ?organ
WHERE {{
    ?concept rdf:type skos:Concept ;
             skos:prefLabel ?prefLabel ;
             skos:broader ?os .
    ?os rdfs:label ?organ .
    FILTER(CONTAINS(LCASE(str(?prefLabel)), "hypertension"))
}}
LIMIT 20
""",

    "5. Patients with hypertension via SKOS ontology": f"""
{SPARQL_PREFIXES}
SELECT (COUNT(DISTINCT ?pat) AS ?patient_count)
WHERE {{
    # 1. Find concepts labelled "hypertension"
    ?concept rdf:type skos:Concept ;
             skos:prefLabel ?prefLabel .
    FILTER(CONTAINS(LCASE(str(?prefLabel)), "hypertension"))

    # 2. Match prefLabel to diagnosis problem
    ?diag rdf:type eicu:Diagnosis ;
          eicu:problem ?prefLabel .

    # 3. Find patients with that diagnosis
    ?pat eicu:hasDiagnosis ?diag .
}}
""",

    "6. Hospital drug profile (top hospital by drugs ordered)": f"""
{SPARQL_PREFIXES}
SELECT ?h ?region (COUNT(?drug) AS ?drugs_ordered)
WHERE {{
    ?h rdf:type eicu:Hospital ;
       eicu:region ?region ;
       eicu:ordered ?drug .
}}
GROUP BY ?h ?region
ORDER BY DESC(?drugs_ordered)
LIMIT 5
""",

    "7. Patient case summary (patient 141765)": f"""
{SPARQL_PREFIXES}
SELECT ?prop ?value
WHERE {{
    data:patient/141765 ?prop ?value .
    FILTER(?prop != rdf:type)
}}
ORDER BY ?prop
""",

    "8. Top 10 infusion drugs": f"""
{SPARQL_PREFIXES}
SELECT ?drugLabel (COUNT(?pat) AS ?patients)
WHERE {{
    ?pat eicu:confirmedInfusion ?drug .
    ?drug rdfs:label ?drugLabel .
}}
GROUP BY ?drugLabel
ORDER BY DESC(?patients)
LIMIT 10
""",

    "9. ICU mortality rate by organ system": f"""
{SPARQL_PREFIXES}
SELECT ?organ
       (COUNT(DISTINCT ?pat) AS ?total)
       (SUM(IF(?died = "true"^^xsd:boolean, 1, 0)) AS ?deaths)
WHERE {{
    ?pat rdf:type eicu:Patient ;
         eicu:icuMortality ?died ;
         eicu:hasDiagnosis ?diag .
    ?diag eicu:belongsTo ?os .
    ?os rdfs:label ?organ .
}}
GROUP BY ?organ
ORDER BY DESC(?deaths)
LIMIT 10
""",

    "10. Concepts with ICD-9 cross-mapping": f"""
{SPARQL_PREFIXES}
SELECT ?concept ?prefLabel ?icd9
WHERE {{
    ?concept rdf:type skos:Concept ;
             skos:prefLabel ?prefLabel ;
             skos:exactMatch ?icd9 .
}}
LIMIT 15
""",
}


# == Query execution ========================================================

def run_query(store: Store, sparql: str, label: str = ""):
    """Execute a SPARQL query and print results as a table."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    try:
        results = list(store.query(sparql))
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    if not results:
        print("  (no results)")
        return

    # Determine column names from first result
    if hasattr(results[0], '__getitem__'):
        # QuerySolution - extract variable names from the SPARQL
        import re
        vars_found = re.findall(r'SELECT\s+.*?WHERE', sparql, re.DOTALL | re.IGNORECASE)
        if vars_found:
            var_names = re.findall(r'\?(\w+)', vars_found[0])
        else:
            var_names = [f"col{i}" for i in range(10)]

        # Print header
        header = "  " + " | ".join(f"{v:30s}" for v in var_names)
        print(header)
        print("  " + "-" * len(header))

        for row in results:
            vals = []
            for v in var_names:
                try:
                    cell = row[v]
                    vals.append(str(cell.value) if cell else "")
                except Exception:
                    vals.append("")
            print("  " + " | ".join(f"{val:30s}" for val in vals))
    else:
        # Boolean result
        print(f"  Result: {results}")

    print(f"  ({len(results)} rows)")


# == Main ====================================================================

def main():
    load_dotenv()
    store_path = os.getenv("OXIGRAPH_STORE_PATH", "/N/scratch/alikh/oxigraph_store")

    if not os.path.isdir(store_path):
        print(f"ERROR: Oxigraph store not found at {store_path}")
        print("       Run  python build_kg.py  first to populate the knowledge graph.")
        sys.exit(1)

    store = Store(store_path)
    print(f"✓ Opened Oxigraph store at {store_path}  ({len(store):,} triples)")

    parser = argparse.ArgumentParser(description="SPARQL query interface for eICU KG")
    parser.add_argument("--query", "-q", type=str, help="Execute a single SPARQL query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL")
    args = parser.parse_args()

    if args.query:
        sparql = args.query
        # Auto-add prefixes if not present
        if "PREFIX" not in sparql.upper():
            sparql = SPARQL_PREFIXES + "\n" + sparql
        run_query(store, sparql, "Ad-hoc query")

    elif args.interactive:
        print("\n  Interactive SPARQL REPL  (type 'quit' to exit, 'demos' for demo list)")
        print(f"  Prefixes auto-injected: eicu:, data:, skos:, rdf:, rdfs:, xsd:\n")
        while True:
            try:
                q = input("sparql> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.lower() == "demos":
                for name in DEMO_QUERIES:
                    print(f"  {name}")
                continue
            if q.startswith("#"):
                # Run a numbered demo, e.g. "#3"
                num = q[1:].strip().rstrip(".")
                for name, sparql in DEMO_QUERIES.items():
                    if name.startswith(num + "."):
                        run_query(store, sparql, name)
                        break
                else:
                    print(f"  Demo #{num} not found.")
                continue
            if "PREFIX" not in q.upper():
                q = SPARQL_PREFIXES + "\n" + q
            run_query(store, q, "")

    else:
        # Run all demos
        print("\n  Running all demo queries...\n")
        for name, sparql in DEMO_QUERIES.items():
            run_query(store, sparql, name)


if __name__ == "__main__":
    main()
