#!/usr/bin/env python3
"""
Test imports for the src.cbr package.

Usage (from project root):
    source venv/bin/activate
    python scripts/test_imports.py
"""

import importlib
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path so `src` package can be resolved.
# Project root is assumed to be the parent directory of this `scripts/` folder.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

modules = [
    "src.cbr",
    "src.cbr.case_library",
    "src.cbr.retrieval",
    "src.cbr.adaptation",
    "src.cbr.evaluation",
]

ok = True
for m in modules:
    print(f"Importing {m} ...", end=" ")
    try:
        importlib.import_module(m)
        print("OK")
    except Exception:
        ok = False
        print("FAIL")
        traceback.print_exc()
        print("-" * 60)

if ok:
    print("\nAll imports succeeded.")
    sys.exit(0)
else:
    print("\nSome imports failed. See tracebacks above.")
    sys.exit(2)
