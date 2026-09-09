"""SAAGA Fuzzing Package.
=====================
Adversarial mutation operators, stacked cipher encoding, and near-miss
fallback pipelines for fuzzing LLM defense scenarios.
"""

from __future__ import annotations

from saaga.fuzzing.fallback import (
    DEFAULT_MIN_SCORE_THRESHOLD,
    DEFAULT_MUTATOR_POOL,
    DEFAULT_NUM_VARIANTS,
    MutationFallback,
    MutationFallbackResult,
    run_mutation_fallback,
    run_mutation_fallback_batch,
)
from saaga.fuzzing.mutators import (
    AVAILABLE_MUTATORS,
    apply_mutator,
    encoding_replay,
    get_mutator,
    policy,
    punctuation_insertion,
    random_deletion,
    random_insertion,
    random_replacement,
    synonym_replacement,
    targeted_insertion,
    targeted_replacement,
    translation,
)

__all__ = [
    # Core fallback engine
    "MutationFallback",
    "MutationFallbackResult",
    "run_mutation_fallback",
    "run_mutation_fallback_batch",
    "DEFAULT_MUTATOR_POOL",
    "DEFAULT_NUM_VARIANTS",
    "DEFAULT_MIN_SCORE_THRESHOLD",
    # Mutators
    "AVAILABLE_MUTATORS",
    "apply_mutator",
    "get_mutator",
    "synonym_replacement",
    "punctuation_insertion",
    "translation",
    "encoding_replay",
    "random_replacement",
    "random_insertion",
    "targeted_replacement",
    "targeted_insertion",
    "random_deletion",
    "policy",
]
