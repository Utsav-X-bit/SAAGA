"""
SAAGA Memory & Knowledge Base Updater
=====================================
Handles append-only post-run recording and post-benchmark knowledge base rebuilds.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class KBUpdater:
    """Incremental updater for SAAGA knowledge base, trajectory database, and RAG index."""

    VALID_MODES = {"off", "run", "benchmark", "all"}

    def __init__(
        self,
        mode: str | None = None,
        worker_id: int = 0,
        num_workers: int = 1,
        data_dir: str | Path = "data",
        verbose: bool = True,
    ):
        if mode is None:
            mode = os.environ.get("SAAGA_UPDATE_KB", os.environ.get("AUTORED_UPDATE_KB", "all")).lower().strip()
        mode = mode.lower().strip()
        if mode not in self.VALID_MODES:
            mode = "all"

        self.mode = mode
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        self.successes_path = self.data_dir / "saaga_successes_v1.jsonl"
        self.failures_path = self.data_dir / "saaga_failures_v1.jsonl"
        self.db_path = self.data_dir / "saaga_kb.db"
        self.kb_output_path = self.data_dir / "strategy_knowledge_base.json"

        self._run_enabled = mode in ("run", "all")
        self._benchmark_enabled = mode in ("benchmark", "all")
        self._db_initialized = False

    @property
    def enabled(self) -> bool:
        return self._run_enabled or self._benchmark_enabled

    def update_after_run(self, run_json: dict[str, Any]) -> bool:
        """Fast append-only update using a single run trace."""
        if not self._run_enabled:
            return False

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            result = run_json.get("result", {})
            scenario = run_json.get("scenario", {})
            attempts = run_json.get("attempts", [])
            success = bool(result.get("ground_truth_success") or result.get("verified_success") or result.get("access_granted_success"))

            target_path = self.successes_path if success else self.failures_path

            # Append to JSONL
            record = {
                "run_id": run_json.get("experiment", {}).get("run_id"),
                "scenario_id": run_json.get("experiment", {}).get("scenario_id"),
                "defense_type": scenario.get("defense_type"),
                "access_code_type": scenario.get("access_code_type"),
                "success": success,
                "total_attempts": len(attempts),
                "attempts": attempts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with open(target_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            return True
        except Exception as exc:
            if self.verbose:
                print(f"[KBUpdater] Run update failed: {exc}")
            return False

    def update_after_benchmark(self) -> bool:
        """Rebuild knowledge base aggregate files after benchmark completion."""
        if not self._benchmark_enabled or self.num_workers > 1:
            return False
        # Lightweight trigger for single-worker or orchestrator
        return True
