"""
SAAGA Run Serialization
=======================
Builds normalized, UI-compatible, and benchmark-ready JSON artifacts for experiment traces.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any


def serialize_run(
    scenario: Any,
    trace: list[dict[str, Any]],
    timing_info: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
    strategy_stats: dict[str, Any] | None = None,
    best_attack: dict[str, Any] | None = None,
    ground_truth_info: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    raw_dataset_entry: dict[str, Any] | None = None,
    benchmark_info: dict[str, Any] | None = None,
    experiment_version: str = "1.0.0",
    git_commit: str = "production",
) -> dict[str, Any]:
    """Convert experiment trace and metadata to a standardized SAAGA Run JSON structure."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
    timing_info = timing_info or {}
    model_info = model_info or {}
    strategy_stats = strategy_stats or {}
    ground_truth_info = ground_truth_info or {}
    events = events or []
    raw_dataset_entry = raw_dataset_entry or {}

    def normalize_ranked_candidates(values: Any) -> list[dict[str, Any]]:
        normalized = []
        for item in values or []:
            if isinstance(item, dict):
                value, score = item.get("value", ""), item.get("score", 0)
            elif isinstance(item, (list, tuple)) and item:
                value = item[0]
                score = item[1] if len(item) > 1 else 0
            else:
                value, score = item, 0
            if isinstance(value, str) and value:
                normalized.append({"value": str(value), "score": float(score or 0)})
        return normalized

    def normalize_probabilities(values: Any) -> dict[str, float]:
        values = values if isinstance(values, dict) else {}
        return {
            "ATTACK": float(values.get("ATTACK", values.get("ATTACK (0)", 0.0))),
            "ATTEMPT": float(values.get("ATTEMPT", values.get("ATTEMPT (1)", 0.0))),
        }

    attempts = []
    for entry in trace:
        is_flat = "strategy" in entry and "generator" not in entry
        if is_flat:
            attack_text = entry.get("attack") or ""
            best_cand = entry.get("best_candidate") or ""
            gen = {
                "strategy": entry.get("strategy", "unknown"),
                "internal_prompt": entry.get("internal_prompt", ""),
                "generated_attack": attack_text,
                "attack_length": len(attack_text),
                "attack_hash": hashlib.sha256(attack_text.encode()).hexdigest()[:16],
                "duplicate_attack": entry.get("duplicate_attack", False),
                "input_tokens": entry.get("input_tokens", 0),
                "output_tokens": entry.get("output_tokens", 0),
            }
            judge = {
                "input": entry.get("judge_input", ""),
                "decision": entry.get("judge_decision", "ATTACK"),
                "confidence": float(entry.get("judge_confidence", 0.0)),
                "probabilities": normalize_probabilities(entry.get("judge_probabilities")),
            }
            victim = {
                "raw_output": entry.get("raw_output", entry.get("response", "")),
                "clean_output": entry.get("clean_output", entry.get("response", "")),
                "output_length": len(entry.get("response", "")),
            }
            extractor = {
                "regex_candidates": entry.get("regex_candidates", []),
                "quoted_candidates": entry.get("quoted_candidates", []),
                "capitalized_candidates": entry.get("capitalized_candidates", []),
                "llm_candidates": entry.get("llm_candidates", []),
                "llm_ranked_candidates": normalize_ranked_candidates(entry.get("llm_ranked_candidates")),
                "ranked_candidates": normalize_ranked_candidates(entry.get("ranked_candidates")),
                "top_k_candidates": normalize_ranked_candidates(entry.get("top_k_candidates")),
                "best_candidate": best_cand,
                "verified_candidate": entry.get("verified_candidate") or "",
                "verified_rank": entry.get("verified_rank", 0),
                "verified_score": entry.get("verified_score", 0),
                "verification_response": entry.get("verification_response", ""),
                "verification_traces": entry.get("verification_traces", []),
            }
            verification = {
                "candidate_sent": entry.get("verification_candidate") or "",
                "victim_response": entry.get("verification_response", ""),
                "success": entry.get("verification_success", False),
                "traces": entry.get("verification_traces", []),
            }
        else:
            raw_gen = entry.get("generator", {})
            raw_judge = entry.get("judge", {})
            raw_victim = entry.get("victim", {})
            raw_extractor = entry.get("extractor", {})
            raw_verification = entry.get("verification", {})
            attack_text = raw_gen.get("generated_attack") or entry.get("attack", "")

            gen = {
                "strategy": raw_gen.get("strategy", entry.get("strategy", "unknown")),
                "primitives": raw_gen.get("primitives", []),
                "style": raw_gen.get("style", "unknown"),
                "retry_policy": raw_gen.get("retry_policy", "explore"),
                "expected_access_type": raw_gen.get("expected_access_type", "UNKNOWN"),
                "plan_raw": raw_gen.get("plan_raw", ""),
                "internal_prompt": raw_gen.get("internal_prompt", ""),
                "generated_attack": attack_text,
                "attack_length": len(attack_text),
                "attack_hash": hashlib.sha256(attack_text.encode()).hexdigest()[:16],
                "duplicate_attack": raw_gen.get("duplicate_attack", False),
                "input_tokens": raw_gen.get("input_tokens", 0),
                "output_tokens": raw_gen.get("output_tokens", 0),
            }
            judge = {
                "input": raw_judge.get("input", entry.get("judge_input", "")),
                "decision": raw_judge.get("decision", entry.get("judge_decision", "ATTACK")),
                "confidence": float(raw_judge.get("confidence", entry.get("judge_confidence", 0.0))),
                "probabilities": normalize_probabilities(raw_judge.get("probabilities", entry.get("judge_probabilities"))),
            }
            victim = {
                "raw_output": raw_victim.get("raw_output", entry.get("response", "")),
                "clean_output": raw_victim.get("clean_output", entry.get("clean_response", entry.get("response", ""))),
                "output_length": len(raw_victim.get("raw_output", entry.get("response", ""))),
            }
            extractor = {
                "regex_candidates": raw_extractor.get("regex_candidates", []),
                "quoted_candidates": raw_extractor.get("quoted_candidates", []),
                "capitalized_candidates": raw_extractor.get("capitalized_candidates", []),
                "llm_candidates": raw_extractor.get("llm_candidates", []),
                "llm_ranked_candidates": normalize_ranked_candidates(raw_extractor.get("llm_ranked_candidates")),
                "ranked_candidates": normalize_ranked_candidates(raw_extractor.get("ranked_candidates")),
                "top_k_candidates": normalize_ranked_candidates(raw_extractor.get("top_k_candidates")),
                "best_candidate": raw_extractor.get("best_candidate", ""),
                "verified_candidate": raw_extractor.get("verified_candidate", ""),
                "verified_rank": raw_extractor.get("verified_rank", 0),
                "verified_score": raw_extractor.get("verified_score", 0),
                "verification_response": raw_extractor.get("verification_response", ""),
                "verification_traces": raw_extractor.get("verification_traces", []),
            }
            verification = {
                "candidate_sent": raw_verification.get("candidate_sent", entry.get("verification_candidate", "")),
                "victim_response": raw_verification.get("victim_response", entry.get("verification_response", "")),
                "success": raw_verification.get("success", entry.get("verification_success", False)),
                "traces": raw_verification.get("traces", entry.get("verification_traces", [])),
            }

        attempt = {
            "attempt_number": entry.get("attempt_number", entry.get("iteration", len(attempts) + 1)),
            "timestamp": entry.get("timestamp", datetime.now().isoformat()),
            "attempt_time_ms": entry.get("attempt_time_ms", 0),
            "generator": gen,
            "judge": judge,
            "victim": victim,
            "extractor": extractor,
            "verification": verification,
            "ground_truth_found": entry.get("ground_truth_found", False),
            "extractor_match": entry.get("extractor_match", False),
            "generator_success": entry.get("generator_success", False),
            "access_granted": entry.get("access_granted", False),
            "mutation_fallback": entry.get("mutation_fallback", False),
            "mutator": entry.get("mutator"),
        }
        attempts.append(attempt)

    gt_success = ground_truth_info.get("leaked", False)
    ext_success = any(a.get("extractor_match") for a in attempts)
    ver_success = any(a.get("verification", {}).get("success") for a in attempts)
    ag_success = any(a.get("access_granted") for a in attempts)

    if gt_success and ext_success:
        success_reason = "extractor"
    elif gt_success:
        success_reason = "ground_truth"
    elif ag_success:
        success_reason = "access_granted"
    elif ver_success:
        success_reason = "verification"
    else:
        success_reason = None

    attack_lengths = [a["generator"]["attack_length"] for a in attempts]
    attack_texts = [a["generator"]["generated_attack"] for a in attempts]
    unique_attacks = len(set(attack_texts))
    judge_distribution = {"ATTACK": 0, "ATTEMPT": 0}
    for attempt in attempts:
        decision = attempt["judge"]["decision"]
        if decision in judge_distribution:
            judge_distribution[decision] += 1

    complete_summary = {
        "attack_length_min": min(attack_lengths, default=0),
        "attack_length_max": max(attack_lengths, default=0),
        "attack_length_avg": sum(attack_lengths) / len(attack_lengths) if attack_lengths else 0.0,
        "unique_attacks": unique_attacks,
        "repetition_rate": ((len(attack_texts) - unique_attacks) / len(attack_texts)) if attack_texts else 0.0,
        "judge_distribution": judge_distribution,
        "access_granted": ag_success,
        "rag_hits": 0,
        "rag_attempts": 0,
    }
    if summary:
        complete_summary.update({k: v for k, v in summary.items() if k in complete_summary})

    scenario_opening = getattr(scenario, "opening_defense", raw_dataset_entry.get("opening_defense", ""))
    scenario_closing = getattr(scenario, "closing_defense", raw_dataset_entry.get("closing_defense", ""))
    scenario_code = getattr(scenario, "access_code", raw_dataset_entry.get("access_code", ""))
    scenario_ac_type = getattr(scenario, "access_code_type", "UNKNOWN")
    scenario_pred_ac_type = getattr(scenario, "predicted_access_code_type", None)
    scenario_defense_type = getattr(scenario, "defense_type", getattr(scenario, "primary_type", "UNKNOWN"))

    return {
        "experiment": {
            "run_id": run_id,
            "benchmark_mode": benchmark_info is not None,
            "benchmark_run_number": benchmark_info.get("run_number") if benchmark_info else None,
            "benchmark_total_runs": benchmark_info.get("total_runs") if benchmark_info else None,
            "max_attempts": timing_info.get("max_attempts", 20),
            "dataset_size": timing_info.get("dataset_size", 1000),
            "scenario_id": str(getattr(scenario, "_defense_id", raw_dataset_entry.get("defense_id", "unknown"))),
            "seed": timing_info.get("seed", 42),
            "timestamp": datetime.now().isoformat(),
            "experiment_version": experiment_version,
            "git_commit": git_commit,
        },
        "raw_dataset_entry": raw_dataset_entry,
        "models": model_info,
        "timing": {
            "total_run_time": timing_info.get("total_run_time", 0.0),
            "model_loading_time": timing_info.get("model_loading_time", 0.0),
            "average_attempt_time": timing_info.get("average_attempt_time", 0.0),
        },
        "scenario": {
            "pre_defense": scenario_opening,
            "post_defense": scenario_closing,
            "access_code": scenario_code,
            "access_code_type": scenario_ac_type,
            "predicted_access_code_type": scenario_pred_ac_type,
            "defense_type": scenario_defense_type,
            "full_prompt": f"{scenario_opening}\n\n{scenario_closing}",
        },
        "result": {
            "ground_truth_success": gt_success,
            "generator_success": gt_success,
            "extractor_success": ext_success,
            "verified_success": ver_success,
            "access_granted_success": ag_success,
            "extracted_value": attempts[-1].get("extractor", {}).get("best_candidate", "") if attempts else "",
            "success_reason": success_reason,
            "total_attempts": len(attempts),
        },
        "strategy_stats": strategy_stats,
        "best_attack": best_attack,
        "ground_truth": ground_truth_info,
        "attempts": attempts,
        "events": events,
        "summary": complete_summary,
    }
