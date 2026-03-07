#!/usr/bin/env python3
"""
fetch_diagnosis.py - Retrieve RDF triples for a diagnosis by keyword or URI.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from pyoxigraph import Store, NamedNode

# Load schema parts needed
from rdf_schema import SPARQL_PREFIXES

def run_query(store: Store, sparql: str):
    """Execute SPARQL and return results as list of dicts."""
    try:
        results = list(store.query(sparql))
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        return []

def search_diagnoses(store: Store, keyword: str):
    """Search for diagnosis URIs matching a keyword."""
    sparql = f"""
    {SPARQL_PREFIXES}
    SELECT DISTINCT ?diag ?label
    WHERE {{
        ?diag rdf:type eicu:Diagnosis ;
              eicu:diagnosisString ?label .
        FILTER(CONTAINS(LCASE(?label), LCASE("{keyword}")))
    }}
    LIMIT 10
    """
    return run_query(store, sparql)

def get_diagnosis_triples(store: Store, uri: str):
    """Get all triples where the diagnosis URI is the subject."""
    # Handle prefixed URI or full URI
    target = uri
    if not (uri.startswith("http://") or uri.startswith("<")):
        # Assume it might be a partial data path if it doesn't look like a URI
        if uri.startswith("data:"):
            target = uri.replace("data:", "http://eicu.mit.edu/data/")
        else:
            target = f"http://eicu.mit.edu/data/diagnosis/{uri}"
    
    sparql = f"""
    {SPARQL_PREFIXES}
    SELECT ?p ?o
    WHERE {{
        <{target}> ?p ?o .
    }}
    """
    return run_query(store, sparql)

def main():
    load_dotenv()
    store_path = os.getenv("OXIGRAPH_STORE_PATH", "/N/scratch/alikh/oxigraph_store")
    
    if not os.path.isdir(store_path):
        print(f"Error: Oxigraph store not found at {store_path}")
        sys.exit(1)
        
    store = Store(store_path)
    
    parser = argparse.ArgumentParser(description="Fetch RDF for eICU diagnoses")
    parser.add_argument("--search", "-s", type=str, help="Search for diagnosis by keyword")
    parser.add_argument("--uri", "-u", type=str, help="Fetch triples for a specific diagnosis URI")
    args = parser.parse_args()
    
    if args.search:
        print(f"Searching for diagnoses matching: '{args.search}'")
        results = search_diagnoses(store, args.search)
        if not results:
            print("No matches found.")
        else:
            print(f"{'URI':50} | {'Diagnosis String'}")
            print("-" * 100)
            for row in results:
                print(f"{str(row['diag'].value):50} | {row['label'].value}")
                
    elif args.uri:
        print(f"Fetching triples for: {args.uri}")
        results = get_diagnosis_triples(store, args.uri)
        if not results:
            print("No triples found for this URI.")
        else:
            print(f"{'Predicate':50} | {'Object'}")
            print("-" * 100)
            for row in results:
                print(f"{str(row['p'].value):50} | {row['o'].value}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
