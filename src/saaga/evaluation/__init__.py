"""
SAAGA Defense Evaluation
========================
Scores a victim LLM's defense strength against SAAGA's adaptive red-teaming
framework, stratified by TensorTrust recoverability difficulty tiers.

Public API:
    run_evaluation      — orchestrate static/adaptive evaluation, return scorecard(s).
    compute_scorecard   — build a DefenseScoreCard from tiered run JSON artifacts.
    write_scorecard     — emit summary.json + report.md.
    DifficultyTier      — recoverability difficulty tier metadata.
"""
from saaga.evaluation.defense_scorer import (
    DefenseScoreCard,
    TierStats,
    compute_scorecard,
    compute_tier_stats,
    derive_severity,
    is_broken,
    winning_reason,
)
from saaga.evaluation.difficulty import (
    DEFAULT_TIERS,
    DifficultyTier,
    all_tiers,
    resolve_tier,
    subset_files_for_tiers,
    tier_from_subset_filename,
)
from saaga.evaluation.eval_runner import run_evaluation
from saaga.evaluation.report import render_markdown, write_scorecard

__all__ = [
    "DefenseScoreCard",
    "TierStats",
    "compute_scorecard",
    "compute_tier_stats",
    "derive_severity",
    "is_broken",
    "winning_reason",
    "DEFAULT_TIERS",
    "DifficultyTier",
    "all_tiers",
    "resolve_tier",
    "subset_files_for_tiers",
    "tier_from_subset_filename",
    "run_evaluation",
    "render_markdown",
    "write_scorecard",
]