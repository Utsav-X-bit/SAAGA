"""SAAGA evaluators package.

Provides modular, robust evaluation and extraction components:
- `SensitiveInfoExtractor`: 6-layer sensitive information extraction pipeline.
- `ReplayVerifier`: Standalone replay verification querying victim models with candidate secrets.
- `StopPointIdentifier`: DistilBERT binary judge with lazy loading and heuristic fallback.
- `AccessCodePredictor`: DistilBERT 4-class secret shape classifier with heuristic fallback.
- `DefenseClassifier`: DistilBERT 8-class prompt defense taxonomy classifier with heuristic fallback.
"""
from __future__ import annotations

from saaga.evaluators.extractor import (
    EXTRACTOR_PATTERNS,
    QUOTED_PATTERNS,
    SensitiveInfoExtractor,
)
from saaga.evaluators.judge import (
    DecisionType,
    StopPointIdentifier,
)
from saaga.evaluators.shape_predictor import (
    AccessCodePredictor,
    AccessCodeType,
    heuristic_categorize_code,
)
from saaga.evaluators.defense_classifier import (
    DefenseClassifier,
    DefenseType,
)
from saaga.evaluators.verifier import (
    ReplayVerifier,
    VerificationResult,
    check_access_granted,
    normalize_candidate_key,
)

__all__ = [
    "EXTRACTOR_PATTERNS",
    "QUOTED_PATTERNS",
    "SensitiveInfoExtractor",
    "ReplayVerifier",
    "VerificationResult",
    "check_access_granted",
    "normalize_candidate_key",
    "StopPointIdentifier",
    "DecisionType",
    "AccessCodePredictor",
    "AccessCodeType",
    "heuristic_categorize_code",
    "DefenseClassifier",
    "DefenseType",
]
