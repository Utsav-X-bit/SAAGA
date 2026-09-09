"""
SAAGA Engine Benchmark Orchestrator
===================================
Drives dataset benchmarking, worker sharding, metrics calculation, and summary reporting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from saaga.core.scenario import DefenseScenario
from saaga.engine.batch_runner import run_scenarios_batched
from saaga.providers.base import BaseLLMProvider
from saaga.reporting.layout import run_filename, runs_root

logger = logging.getLogger(__name__)


def load_scenarios_from_jsonl(
    dataset_path: str | Path,
    limit: Optional[int] = None,
    start_idx: int = 0,
) -> list[DefenseScenario]:
    """Load defense scenarios from a JSONL file."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    scenarios = []
    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if limit is not None and len(scenarios) >= limit:
                break
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                sc = DefenseScenario(
                    opening_defense=item.get("opening_defense", ""),
                    closing_defense=item.get("closing_defense", ""),
                    access_code=item.get("access_code", ""),
                    access_code_type=item.get("access_code_type", "UNKNOWN"),
                    defense_type=item.get("defense_type", "UNKNOWN"),
                )
                sc._defense_id = str(item.get("defense_id", idx))
                scenarios.append(sc)
            except Exception as exc:
                logger.warning("Failed to parse JSONL line %d: %s", idx, exc)
    return scenarios


def execute_benchmark(
    scenarios: Sequence[DefenseScenario],
    victim_provider: BaseLLMProvider,
    output_dir: str | Path = "results/benchmark",
    max_attempts: int = 20,
    enable_fallback: bool = True,
    batch_size: int = 16,
    worker_id: int = 0,
) -> dict[str, Any]:
    """Execute a benchmark over a list of scenarios and emit standard output layout."""
    total = len(scenarios)
    out_root = runs_root(output_dir, "benchmark", victim_provider.model_id, f"{total}rounds")

    all_runs = []
    total_wins = 0

    # Process in batches
    for i in range(0, total, batch_size):
        chunk = scenarios[i : i + batch_size]
        runs = run_scenarios_batched(
            scenarios=chunk,
            victim_provider=victim_provider,
            planner_provider=victim_provider,
            generator_provider=victim_provider,
            max_attempts=max_attempts,
            enable_fallback=enable_fallback,
        )

        for j, run_json in enumerate(runs):
            sc_idx = i + j + 1
            sc = chunk[j]
            is_win = run_json.get("result", {}).get("success", False)
            if is_win:
                total_wins += 1

            sub = "success" if is_win else "failed"
            fname = run_filename(sc._defense_id, worker_id=worker_id, round_num=sc_idx)
            save_path = out_root / "runs" / sub / fname
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(run_json, f, indent=2)

            all_runs.append(run_json)

    summary = {
        "total_rounds": total,
        "total_successes": total_wins,
        "success_rate": total_wins / total if total else 0.0,
        "victim_model": victim_provider.model_id,
        "worker_id": worker_id,
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
