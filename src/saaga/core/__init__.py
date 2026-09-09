"""
SAAGA Core: Fundamental contracts, scenarios, taxonomies, and scoring mechanisms.
================================================================================
"""
from __future__ import annotations

from saaga.core.constants import (
    ATTACK_TYPE_PROMPTS,
    ATTACK_TYPES,
    DEFAULT_MUTATOR_POOL,
    KNOWN_ACCESS_TYPES,
    KNOWN_FAILURE_REASONS,
    KNOWN_POLICIES,
    KNOWN_STRATEGIES,
    KNOWN_STYLES,
    PLANNER_STUCK_THRESHOLD,
    REFUSAL_KEYWORDS,
    REFUSAL_PHRASES,
    STRATEGY_MUTATOR_MAP,
)
from saaga.core.contract import (
    canonicalize_plan,
    extract_tag,
    normalize_plan_dict,
    parse_plan_text,
    render_plan_xml,
    validate_plan,
)
from saaga.core.scenario import DefenseScenario, categorize_defense_detailed
from saaga.core.scoring import (
    SuccessOutcome,
    classify_failure_mode,
    classify_success,
    compute_fallback_score,
    cooperation_score,
    infer_strategy_from_content,
    resolve_mutator_pool,
    resolve_mutator_pool_cooperative,
    strip_think_blocks,
)

__all__ = [
    # Constants & Taxonomy
    "ATTACK_TYPES",
    "KNOWN_STRATEGIES",
    "KNOWN_STYLES",
    "KNOWN_POLICIES",
    "KNOWN_ACCESS_TYPES",
    "KNOWN_FAILURE_REASONS",
    "REFUSAL_KEYWORDS",
    "REFUSAL_PHRASES",
    "DEFAULT_MUTATOR_POOL",
    "STRATEGY_MUTATOR_MAP",
    "ATTACK_TYPE_PROMPTS",
    "PLANNER_STUCK_THRESHOLD",
    # Scenario
    "DefenseScenario",
    "categorize_defense_detailed",
    # Contract
    "extract_tag",
    "parse_plan_text",
    "normalize_plan_dict",
    "canonicalize_plan",
    "validate_plan",
    "render_plan_xml",
    # Scoring
    "SuccessOutcome",
    "classify_success",
    "classify_failure_mode",
    "cooperation_score",
    "compute_fallback_score",
    "strip_think_blocks",
    "infer_strategy_from_content",
    "resolve_mutator_pool",
    "resolve_mutator_pool_cooperative",
]
