"""AccessCodePredictor: DistilBERT 4-class classifier for secret shape prediction.

Predicts whether a target secret/access code is a:
- TOKEN: single word, password, or short alphanumeric token
- PHRASE: 2 to 5 words
- SENTENCE: full sentence (>5 words or >20 characters)
- MULTILINE: multiline block or long text (>100 characters or containing newlines)

Provides graceful heuristic fallback when model weights are absent.
"""
from __future__ import annotations

import logging
import os
import re
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_ACP_MODEL_PATH = "experiment/access_code_predictor"
FALLBACK_BASE_MODEL = "distilbert-base-uncased"

LABEL_MAP: dict[int, str] = {
    0: "TOKEN",
    1: "MULTILINE",
    2: "PHRASE",
    3: "SENTENCE",
}
INV_LABEL_MAP: dict[str, int] = {v: k for k, v in LABEL_MAP.items()}


class AccessCodeType(str, Enum):
    """Shape category of an access code or secret."""

    TOKEN = "TOKEN"
    MULTILINE = "MULTILINE"
    PHRASE = "PHRASE"
    SENTENCE = "SENTENCE"


def heuristic_categorize_code(access_code: str) -> str:
    """Categorize an access code string into one of the 4 shape categories."""
    if not access_code:
        return AccessCodeType.TOKEN.value

    ac = access_code.strip()
    words = ac.split()

    if "\n" in ac or len(ac) > 100:
        return AccessCodeType.MULTILINE.value
    if len(words) > 5 or len(ac) > 20:
        return AccessCodeType.SENTENCE.value
    if 2 <= len(words) <= 5:
        return AccessCodeType.PHRASE.value
    return AccessCodeType.TOKEN.value


class AccessCodePredictor:
    """DistilBERT 4-class classifier for access code shape prediction with heuristic fallback.

    Features:
    - Lazy loading: model loaded on demand.
    - CPU/GPU device selection.
    - Softmax with temperature scaling when configured.
    - Structural heuristic fallback inspecting access code length or defense prompts when weights are absent.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        fallback_heuristic: bool = True,
    ):
        """Initialize AccessCodePredictor.

        Parameters:
            model: Pre-loaded PyTorch model (optional).
            tokenizer: Pre-loaded HuggingFace tokenizer (optional).
            model_path: Path to model directory or HuggingFace repo.
            device: 'cuda', 'cpu', or None (auto-detect).
            fallback_heuristic: Whether to use heuristic fallback when weights are missing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path or os.environ.get("SAAGA_ACP_MODEL", DEFAULT_ACP_MODEL_PATH)
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
        """Attempt to load the access code predictor model and tokenizer.

        Returns True if successful, False if falling back to heuristic.
        """
        self._load_attempted = True
        path = model_path or self.model_path

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            logger.info(f"PyTorch or Transformers not installed ({e}); using heuristic shape predictor")
            self._is_loaded = False
            return False

        if not path or not os.path.exists(path):
            logger.info(
                f"Access code predictor checkpoint not found at '{path}'; "
                f"using heuristic shape predictor"
            )
            self._is_loaded = False
            return False

        try:
            logger.info(f"Loading Access Code Predictor from {path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            except OSError:
                logger.info(f"Tokenizer not found in {path}; falling back to {FALLBACK_BASE_MODEL}")
                self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_BASE_MODEL)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, num_labels=4, local_files_only=True
            )
            self.model.to(torch.device(self.device_str))
            self.model.eval()
            self._is_loaded = True
            logger.info(f"✓ Access Code Predictor loaded successfully on {self.device_str}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Access Code Predictor from '{path}': {e}; using fallback heuristic")
            self.model = None
            self.tokenizer = None
            self._is_loaded = False
            return False

    def _ensure_loaded(self):
        if not self._is_loaded and not self._load_attempted:
            self.load_model()

    def _extract_defense_text(self, scenario: Any) -> tuple[str, str, Optional[str]]:
        """Extract opening_defense, closing_defense, and access_code from scenario."""
        if scenario is None:
            return "", "", None

        if isinstance(scenario, dict):
            return (
                str(scenario.get("opening_defense", "")),
                str(scenario.get("closing_defense", "")),
                scenario.get("access_code"),
            )

        opening = str(getattr(scenario, "opening_defense", "") or "")
        closing = str(getattr(scenario, "closing_defense", "") or "")
        code = getattr(scenario, "access_code", None)
        return opening, closing, code

    def _heuristic_predict(self, scenario: Any) -> dict[str, float]:
        """Compute shape probabilities using structural heuristics."""
        opening, closing, code = self._extract_defense_text(scenario)

        # 1. If ground truth access code is present, compute near-deterministic distribution
        if code and isinstance(code, str) and code.strip():
            inferred = heuristic_categorize_code(code)
            probs = {
                AccessCodeType.TOKEN.value: 0.05,
                AccessCodeType.MULTILINE.value: 0.05,
                AccessCodeType.PHRASE.value: 0.05,
                AccessCodeType.SENTENCE.value: 0.05,
            }
            probs[inferred] = 0.85
            return probs

        # 2. Inspect defense prompt cues
        combined_text = f"{opening} {closing}".lower()

        if re.search(r"\b(multiline|multiple lines|lines of text|code block|paragraph)\b", combined_text):
            return {
                AccessCodeType.TOKEN.value: 0.10,
                AccessCodeType.MULTILINE.value: 0.65,
                AccessCodeType.PHRASE.value: 0.10,
                AccessCodeType.SENTENCE.value: 0.15,
            }

        if re.search(r"\b(sentence|full sentence|complete sentence|statement)\b", combined_text):
            return {
                AccessCodeType.TOKEN.value: 0.10,
                AccessCodeType.MULTILINE.value: 0.10,
                AccessCodeType.PHRASE.value: 0.20,
                AccessCodeType.SENTENCE.value: 0.60,
            }

        if re.search(r"\b(phrase|keyphrase|passphrase|words|two words|three words)\b", combined_text):
            return {
                AccessCodeType.TOKEN.value: 0.15,
                AccessCodeType.MULTILINE.value: 0.05,
                AccessCodeType.PHRASE.value: 0.65,
                AccessCodeType.SENTENCE.value: 0.15,
            }

        if re.search(r"\b(word|single word|password|token|code|pin)\b", combined_text):
            return {
                AccessCodeType.TOKEN.value: 0.65,
                AccessCodeType.MULTILINE.value: 0.05,
                AccessCodeType.PHRASE.value: 0.20,
                AccessCodeType.SENTENCE.value: 0.10,
            }

        # 3. Default uniform distribution
        return {
            AccessCodeType.TOKEN.value: 0.25,
            AccessCodeType.MULTILINE.value: 0.25,
            AccessCodeType.PHRASE.value: 0.25,
            AccessCodeType.SENTENCE.value: 0.25,
        }

    def predict(self, scenario: Any) -> dict[str, float]:
        """Predict probability distribution over secret shape categories.

        Parameters:
            scenario: DefenseScenario object, dict, or object with opening/closing defenses.

        Returns:
            dict mapping 'TOKEN', 'MULTILINE', 'PHRASE', 'SENTENCE' to probability floats.
        """
        self._ensure_loaded()

        if not self.is_loaded:
            return self._heuristic_predict(scenario)

        opening, closing, _ = self._extract_defense_text(scenario)
        text = f"{opening} [SEP] {closing}"

        import torch

        try:
            device = torch.device(self.device_str)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                temp = getattr(self.model.config, "temperature", 1.0)
                logits = outputs.logits / (temp if temp > 0 else 1.0)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            id2label = getattr(self.model.config, "id2label", None) or LABEL_MAP
            return {
                str(id2label.get(i, LABEL_MAP.get(i, f"CLASS_{i}"))): float(probs[i])
                for i in range(len(probs))
            }
        except Exception as e:
            logger.warning(f"Inference error in AccessCodePredictor ({e}); using heuristic fallback")
            return self._heuristic_predict(scenario)

    def predict_text(self, opening_defense: str, closing_defense: str = "") -> dict[str, float]:
        """Predict shape probabilities from raw defense strings."""
        return self.predict({"opening_defense": opening_defense, "closing_defense": closing_defense})

    def predict_type(self, scenario: Any) -> str:
        """Predict the top access code type label (argmax)."""
        probs = self.predict(scenario)
        return max(probs, key=probs.get)
