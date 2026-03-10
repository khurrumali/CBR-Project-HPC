"""
case_library.py — Persistent Case Library for src.cbr

Provides a lightweight SQLite-backed store for CBR cases.
Each case is keyed by a string case_id and stores:
  - problem  : JSON-serialisable dict (the unsolved problem)
  - solution : JSON-serialisable dict (the adapted solution)
  - outcome  : JSON-serialisable dict, optional (post-hoc evaluation)

Design notes
------------
- Uses a single SQLite file in the user's home scratch dir by default.
- The path can be overridden via the ``CASE_LIBRARY_PATH`` env-var or by
  passing ``db_path`` to the constructor.
- All reads/writes are transactional; the file is opened per-operation to
  stay safe in multi-process Slurm environments.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

_DEFAULT_DB = os.path.join(
    os.getenv("SCRATCH", os.path.expanduser("~")),
    "cbr_case_library.sqlite",
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cases (
    case_id  TEXT PRIMARY KEY,
    problem  TEXT NOT NULL,
    solution TEXT NOT NULL,
    outcome  TEXT
)
"""


class CaseLibrary:
    """
    Persistent, SQLite-backed case library.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite file.  Defaults to the value of the
        ``CASE_LIBRARY_PATH`` environment variable, or a file named
        ``cbr_case_library.sqlite`` in ``$SCRATCH`` (falling back to ``~``).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path: str = (
            db_path
            or os.getenv("CASE_LIBRARY_PATH")
            or _DEFAULT_DB
        )
        self._init_db()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> str:
        return self._db_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the cases table if it doesn't exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        case_id: str,
        problem: Dict[str, Any],
        solution: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert or replace a case.

        Parameters
        ----------
        case_id : str
            Unique identifier (e.g. a ``patientunitstayid``).
        problem : dict
            The unsolved-problem representation.
        solution : dict
            The adapted solution.
        outcome : dict, optional
            Post-hoc evaluation metadata.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cases (case_id, problem, solution, outcome)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(case_id),
                    json.dumps(problem, ensure_ascii=False),
                    json.dumps(solution, ensure_ascii=False),
                    json.dumps(outcome, ensure_ascii=False) if outcome is not None else None,
                ),
            )

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a case by id, or ``None`` if not found.

        Returns
        -------
        dict with keys ``case_id``, ``problem``, ``solution``, ``outcome``.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (str(case_id),)
            ).fetchone()
        if row is None:
            return None
        return {
            "case_id": row["case_id"],
            "problem": json.loads(row["problem"]),
            "solution": json.loads(row["solution"]),
            "outcome": json.loads(row["outcome"]) if row["outcome"] else None,
        }

    def remove(self, case_id: str) -> bool:
        """
        Delete a case by id.

        Returns
        -------
        bool
            True if a row was deleted, False if the case_id was not found.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cases WHERE case_id = ?", (str(case_id),)
            )
        return cur.rowcount > 0

    def count(self) -> int:
        """Return the total number of cases stored."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM cases").fetchone()
        return row[0]

    def all_ids(self) -> List[str]:
        """Return a list of all case_ids."""
        with self._connect() as conn:
            rows = conn.execute("SELECT case_id FROM cases ORDER BY case_id").fetchall()
        return [r[0] for r in rows]

    def all_cases(self) -> List[Dict[str, Any]]:
        """Return all cases as a list of dicts."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cases ORDER BY case_id").fetchall()
        return [
            {
                "case_id": r["case_id"],
                "problem": json.loads(r["problem"]),
                "solution": json.loads(r["solution"]),
                "outcome": json.loads(r["outcome"]) if r["outcome"] else None,
            }
            for r in rows
        ]

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"CaseLibrary(db_path={self._db_path!r}, count={self.count()})"
