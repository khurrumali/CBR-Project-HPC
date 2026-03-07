"""
src.cbr — Case-Based Reasoning engine for the eICU clinical project.

This package implements a lightweight CBR (Case-Based Reasoning) surface
for the repository. It provides a thin, import-safe package marker and
convenience re-exports for commonly-used symbols.

Design goals
------------
- Keep import-time side-effects minimal so `import src.cbr` is safe in
  environments where optional dependencies (e.g., torch, pyoxigraph) may
  not be installed.
- Encourage explicit submodule imports for heavier functionality:
    from src.cbr.case_library import CaseLibrary
    from src.cbr.retrieval import retrieve
    from src.cbr.adaptation import substitution_adapt
    from src.cbr.evaluation import retain_case
"""

# Public package version
__version__ = "0.1.0"

# Re-export commonly used names when possible; swallow exceptions to avoid
# causing import-time failures in environments missing optional libs.
__all__ = [
    "CaseLibrary",
    "retrieve",
    "compute_similarity",
    "SimilarityResult",
    "null_adapt",
    "substitution_adapt",
    "rule_based_adapt",
    "llm_adapt",
    "record_outcome",
    "should_retain",
    "retain_case",
]

# Attempt to import and re-export selected symbols for convenience.
# If any import fails, the package still imports cleanly and callers can
# import the submodules directly for full tracebacks.
try:
    from .case_library import CaseLibrary  # type: ignore
except Exception:
    CaseLibrary = None  # type: ignore

try:
    from .retrieval import (  # type: ignore
        SimilarityResult,
        compute_similarity,
        retrieve,
    )
except Exception:
    retrieve = None  # type: ignore
    compute_similarity = None  # type: ignore
    SimilarityResult = None  # type: ignore

try:
    from .adaptation import llm_adapt, null_adapt, rule_based_adapt, substitution_adapt  # type: ignore
except Exception:
    null_adapt = None  # type: ignore
    substitution_adapt = None  # type: ignore
    rule_based_adapt = None  # type: ignore
    llm_adapt = None  # type: ignore

try:
    from .evaluation import record_outcome, retain_case, should_retain  # type: ignore
except Exception:

    record_outcome = None  # type: ignore
    should_retain = None  # type: ignore
    retain_case = None  # type: ignore

# Helpful message for interactive use
def _about():
    return (
        "src.cbr — Case-Based Reasoning package (version: {})\n"
        "Modules available: case_library, retrieval, adaptation, evaluation\n"
        "Import submodules directly for full functionality: e.g. "
        "'from src.cbr.case_library import CaseLibrary'".format(__version__)
    )
