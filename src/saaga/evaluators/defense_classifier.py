"""
DefenseClassifier: DistilBERT 8-Class Classifier for Defense Prompt Taxonomy.
=============================================================================
Predicts the primary defense mechanism protecting an LLM prompt:
- conditional: logic gates / if-then rule branching
- conversation: dialogue framing, conversational deflection
- exception: absolute bans, strict refusal mandates ("no matter what")
- instruction_hiding: system prompt masking, anti-leak instructions
- password: secret token, PIN, or passphrase verification requirements
- roleplay: persona imposition, character acting, simulated environment
- translation: multilingual encoding, foreign language translation defense
- trigger_phrase: exact magic words / output phrase matching

Provides graceful heuristic fallback to regex/keyword rules when weights are absent.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from saaga.core.scenario import categorize_defense_detailed

logger = logging.getLogger(__name__)

DEFAULT_DEFENSE_CLASSIFIER_PATH = os.environ.get(
    "SAAGA_DEFENSE_CLASSIFIER_MODEL", "models/defense_classifier"
)
FALLBACK_BASE_MODEL = "distilbert-base-uncased"

LABEL_MAP: dict[int, str] = {
    0: "conditional",
    1: "conversation",
    2: "exception",
    3: "instruction_hiding",
    4: "password",
    5: "roleplay",
    6: "translation",
    7: "trigger_phrase",
}
INV_LABEL_MAP: dict[str, int] = {v: k for k, v in LABEL_MAP.items()}


class DefenseType(str, Enum):
    """Canonical taxonomy classes for prompt defense mechanisms."""

    CONDITIONAL = "conditional"
    CONVERSATION = "conversation"
    EXCEPTION = "exception"
    INSTRUCTION_HIDING = "instruction_hiding"
    PASSWORD = "password"
    ROLEPLAY = "roleplay"
    TRANSLATION = "translation"
    TRIGGER_PHRASE = "trigger_phrase"


class DefenseClassifier:
    """DistilBERT 8-class classifier for prompt defense categorization with heuristic fallback.

    Features:
    - Lazy loading: model loaded on demand.
    - CPU/GPU device selection.
    - Softmax probability distribution over the 8 defense taxonomy types.
    - Structural heuristic fallback inspecting opening/closing prompts when weights are absent.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        fallback_heuristic: bool = True,
    ):
        """Initialize DefenseClassifier.

        Parameters:
            model: Pre-loaded PyTorch model (optional).
            tokenizer: Pre-loaded HuggingFace tokenizer (optional).
            model_path: Path to model directory or HuggingFace repo.
            device: 'cuda', 'cpu', or None (auto-detect).
            fallback_heuristic: Whether to use heuristic fallback when weights are missing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path or DEFAULT_DEFENSE_CLASSIFIER_PATH
        self.fallback_heuristic = fallback_heuristic

        if device:
            self.device_str = device
        else:
            try:
                import torch

                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device_str = "cpu"

        self._is_loaded = self.model is not None and self.tokenizer is not None
        self._load_attempted = False

    @property
    def is_loaded(self) -> bool:
        """Whether the PyTorch model and tokenizer are active."""
        return self._is_loaded and self.model is not None

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Attempt to load the defense classifier model and tokenizer.

        Returns True if successful, False if falling back to heuristic.
        """
        self._load_attempted = True
        path = model_path or self.model_path

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            logger.info(f"PyTorch or Transformers not installed ({e}); using heuristic defense classifier")
            self._is_loaded = False
            return False

        if not path or not os.path.exists(path):
            logger.info(
                f"Defense classifier checkpoint not found at '{path}'; "
                f"using heuristic defense classifier"
            )
            self._is_loaded = False
            return False

        try:
            logger.info(f"Loading DefenseClassifier from {path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            )
            self.model.to(torch.device(self.device_str))
            self.model.eval()
            self._is_loaded = True
            logger.info(f"✓ DefenseClassifier loaded successfully on {self.device_str}")
            return True
        except Exception as e:
            logger.warning(
                f"Could not load DefenseClassifier from {path}: {e}; "
                f"using heuristic defense classifier"
            )
            self.model = None
            self.tokenizer = None
            self._is_loaded = False
            return False

    def _ensure_loaded(self) -> None:
        """Trigger lazy loading on first inference request."""
        if not self._is_loaded and not self._load_attempted:
            self.load_model()

    @staticmethod
    def _extract_defense_text(scenario: Any) -> tuple[str, str, str]:
        """Extract opening defense, closing defense, and combined text from diverse object types."""
        if hasattr(scenario, "opening_defense") and hasattr(scenario, "closing_defense"):
            opening = getattr(scenario, "opening_defense", "") or ""
            closing = getattr(scenario, "closing_defense", "") or ""
        elif isinstance(scenario, dict):
            opening = scenario.get("opening_defense", scenario.get("opening", "")) or ""
            closing = scenario.get("closing_defense", scenario.get("closing", "")) or ""
        elif isinstance(scenario, str):
            opening = scenario
            closing = ""
        else:
            opening = str(scenario)
            closing = ""

        combined = f"{opening}\n\n{closing}".strip()
        return opening, closing, combined

    def _heuristic_predict(self, scenario: Any) -> dict[str, float]:
        """Heuristic probability distribution when PyTorch model is unavailable."""
        opening, closing, _ = self._extract_defense_text(scenario)
        primary, secondary = categorize_defense_detailed(opening, closing)

        # Base uniform prior across all 8 classes
        probs = {label: 0.05 for label in LABEL_MAP.values()}

        # Allocate dominant mass to primary classification
        if primary in probs:
            probs[primary] = 0.60

        # Distribute secondary mass
        if secondary:
            sec_weight = 0.25 / len(secondary)
            for s in secondary:
                if s in probs:
                    probs[s] = probs.get(s, 0.0) + sec_weight

        # Re-normalize to sum to 1.0
        total = sum(probs.values())
        return {k: round(v / total, 4) for k, v in probs.items()}

    def predict(self, scenario: Any) -> dict[str, float]:
        """Predict probability distribution over the 8 defense taxonomy categories.

        Parameters:
            scenario: DefenseScenario object, dict, or defense prompt string.

        Returns:
            dict mapping defense taxonomy strings to probability floats.
        """
        self._ensure_loaded()

        if not self.is_loaded:
            return self._heuristic_predict(scenario)

        opening, closing, _ = self._extract_defense_text(scenario)
        text = f"{opening} [SEP] {closing}".strip()

        import torch

        try:
            device = torch.device(self.device_str)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            id2label = getattr(self.model.config, "id2label", None) or LABEL_MAP
            return {
                str(id2label.get(i, LABEL_MAP.get(i, f"CLASS_{i}"))): float(probs[i])
                for i in range(len(probs))
            }
        except Exception as e:
            logger.warning(f"Inference error in DefenseClassifier ({e}); using heuristic fallback")
            return self._heuristic_predict(scenario)

    def predict_type(self, scenario: Any) -> str:
        """Predict the top defense taxonomy type label (argmax)."""
        probs = self.predict(scenario)
        return max(probs, key=probs.get)

    def predict_detailed(self, scenario: Any, threshold: float = 0.20) -> tuple[str, list[str]]:
        """Predict primary defense type and secondary flags exceeding confidence threshold.

        Returns:
            (primary_type, [secondary_flags...])
        """
        probs = self.predict(scenario)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        primary = sorted_probs[0][0]
        secondary = [k for k, v in sorted_probs[1:] if v >= threshold]
        return primary, secondary
