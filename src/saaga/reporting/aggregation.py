"""
SAAGA Benchmark Aggregation
===========================
Merges worker-sharded benchmark runs and summaries into a single comprehensive benchmark artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def merge_benchmarks(input_paths: list[str | Path], output_path: str | Path | None = None) -> dict[str, Any]:
    """Merge multiple worker benchmark summaries into a single unified benchmark report."""
    if not input_paths:
        raise ValueError("No input files provided for merge")

    workers = []
    for p in input_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            workers.append(data)

    merged: dict[str, Any] = {
        "total_rounds": sum(w.get("total_rounds", 0) for w in workers),
        "total_successes": sum(w.get("total_successes", 0) for w in workers),
        "total_access_granted": sum(w.get("total_access_granted", 0) for w in workers),
        "verified_success": sum(w.get("verified_success", 0) for w in workers),
        "top1_success": sum(w.get("top1_success", 0) for w in workers),
        "top3_success": sum(w.get("top3_success", 0) for w in workers),
        "top5_success": sum(w.get("top5_success", 0) for w in workers),
        "total_ground_truth_leaks": sum(w.get("total_ground_truth_leaks", 0) for w in workers),
        "mutation_fallback_triggered": sum(w.get("mutation_fallback_triggered", 0) for w in workers),
        "mutation_fallback_successes": sum(w.get("mutation_fallback_successes", 0) for w in workers),
        "worker_summaries": workers,
    }

    # Rates
    total_r = merged["total_rounds"]
    total_s = merged["total_successes"]
    merged["success_rate"] = (total_s / total_r) if total_r > 0 else 0.0
    merged["verified_rate"] = (merged["verified_success"] / total_r) if total_r > 0 else 0.0
    merged["gt_leak_rate"] = (merged["total_ground_truth_leaks"] / total_r) if total_r > 0 else 0.0

    # Average attempts on success
    succ_attempts_sum = sum(w.get("avg_attempts_on_success", 0.0) * w.get("total_successes", 0) for w in workers)
    merged["avg_attempts_on_success"] = (succ_attempts_sum / total_s) if total_s > 0 else 0.0

    # Propagate run config metadata
    first_worker = workers[0] if workers else {}
    first_meta = first_worker.get("metadata", {}) or first_worker.get("run_config", {})
    merged["seed"] = first_meta.get("seed", first_worker.get("seed", 42))
    merged["dataset_size"] = first_meta.get("dataset_size", first_worker.get("dataset_size", total_r))
    merged["victim_model"] = first_meta.get("victim_model", first_worker.get("victim_model", "unknown"))

    # Fallback diagnostics aggregation
    fb_diag = {"variant_total": 0, "no_op_total": 0, "no_op_rate": 0.0, "mutator_counts": {}}
    for w in workers:
        diag = w.get("mutation_fallback_diagnostics", {})
        fb_diag["variant_total"] += diag.get("variant_total", 0)
        fb_diag["no_op_total"] += diag.get("no_op_total", 0)
        for m, cnt in diag.get("mutator_counts", {}).items():
            fb_diag["mutator_counts"][m] = fb_diag["mutator_counts"].get(m, 0) + cnt

    if fb_diag["variant_total"] > 0:
        fb_diag["no_op_rate"] = fb_diag["no_op_total"] / fb_diag["variant_total"]
    merged["mutation_fallback_diagnostics"] = fb_diag

    # Write output if requested
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    return merged
