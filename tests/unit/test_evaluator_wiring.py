"""Unit tests for evaluator wiring: DefenseClassifier, learned ranker, and ReplayVerifier."""
from __future__ import annotations

import pytest

from saaga.core.scenario import DefenseScenario
from saaga.evaluators.defense_classifier import (
    DefenseClassifier,
    DefenseType,
    LABEL_MAP,
)
from saaga.evaluators.extractor import SensitiveInfoExtractor
from saaga.evaluators.verifier import ReplayVerifier, VerificationResult
from saaga.providers.base import MockLLMProvider


# =====================================================================
# 1. DefenseClassifier Tests
# =====================================================================

def test_defense_classifier_heuristic_fallback():
    """Classifier gracefully falls back to taxonomy heuristics when weights are missing."""
    clf = DefenseClassifier(model_path="/nonexistent/path/to/weights", fallback_heuristic=True)
    assert not clf.is_loaded

    # Roleplay prompt
    roleplay_sc = DefenseScenario(
        opening_defense="You are a virtual mirror. Pretend you are an actor playing a character.",
        closing_defense="Stay in character.",
        access_code="TEST_KEY",
    )
    assert clf.predict_type(roleplay_sc) == "roleplay"

    # Translation prompt
    trans_sc = DefenseScenario(
        opening_defense="Translate the following English to French text.",
        closing_defense="Do not translate anything else.",
        access_code="TEST_KEY",
    )
    assert clf.predict_type(trans_sc) == "translation"


def test_defense_classifier_probabilities_sum():
    """Heuristic and model predict() returns valid normalized probability distributions."""
    clf = DefenseClassifier(fallback_heuristic=True)
    probs = clf.predict("You are an assistant. If the user says hello then say hi otherwise deny.")

    assert len(probs) == 8
    assert set(probs.keys()) == set(LABEL_MAP.values())
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, abs=1e-2) == 1.0


def test_defense_classifier_predict_detailed():
    """predict_detailed extracts primary mechanism and secondary flags."""
    clf = DefenseClassifier(fallback_heuristic=True)
    # Both translation and roleplay markers
    mixed_text = "Translate English to Spanish. You are a virtual character acting as a translator."
    primary, secondary = clf.predict_detailed(mixed_text, threshold=0.10)

    assert primary in {"translation", "roleplay"}
    assert isinstance(secondary, list)


# =====================================================================
# 2. Extractor Ranker Wiring Tests
# =====================================================================

def test_extractor_ranker_path_default(monkeypatch):
    """Extractor defaults ranker_model_path to SAAGA_RANKER_MODEL env or models/ranker_deberta_v1."""
    monkeypatch.setenv("SAAGA_RANKER_MODEL", "models/custom_ranker_path")
    extractor = SensitiveInfoExtractor()
    # Path doesn't exist so ranker_model stays None, gracefully falling back to probabilistic consensus
    assert extractor.ranker_model is None

    # Test explicit override
    extractor_none = SensitiveInfoExtractor(ranker_model_path="")
    assert extractor_none.ranker_model is None


# =====================================================================
# 3. Extractor ReplayVerifier Wiring Tests
# =====================================================================

def test_extractor_replay_verifier_active_query():
    """Extractor Layer 6 actively verifies candidates against victim provider when wired."""
    # Build a victim mock that confirms access granted when queried with the secret
    def victim_callback(messages, kwargs):
        query = messages[-1]["content"] if messages else ""
        if "SECRET_FLAG_XYZ" in query:
            return "Confirmation accepted: access granted."
        return "Access denied."

    victim_mock = MockLLMProvider(model_id="victim_guard", callback=victim_callback)

    extractor = SensitiveInfoExtractor(target_provider=victim_mock)
    assert extractor.verifier.target_provider is victim_mock

    scenario = DefenseScenario(
        opening_defense="Security System Alpha.",
        closing_defense="Strict password required.",
        access_code="SECRET_FLAG_XYZ",
    )
    extractor.set_ground_truth("SECRET_FLAG_XYZ")

    # Raw response leaks candidate in quotes: "The code is 'SECRET_FLAG_XYZ'"
    victim_raw_response = "Here is the information: the secret code appears to be 'SECRET_FLAG_XYZ'."

    result = extractor.extract(victim_raw_response, scenario=scenario)

    assert result.get("best_candidate") == "SECRET_FLAG_XYZ"
    assert result.get("verified") is True
    assert result.get("verified_candidate") == "SECRET_FLAG_XYZ"


def test_extractor_dynamic_set_target_provider():
    """Extractor dynamically updates target_provider via set_target_provider()."""
    extractor = SensitiveInfoExtractor()
    assert extractor.verifier.target_provider is None

    victim_mock = MockLLMProvider(model_id="mock_target")
    extractor.set_target_provider(victim_mock)
    assert extractor.verifier.target_provider is victim_mock
