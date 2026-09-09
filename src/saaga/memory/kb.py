"""
SAAGA Strategy Knowledge Base
==============================
Tracks empirical success rates of attack strategies across defense families and victim models.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


class StrategyKnowledgeBase:
    """Interface for reading and querying the empirical strategy knowledge base."""

    def __init__(self, kb_path: str | Path = "data/strategy_knowledge_base.json"):
        self.kb_path = Path(kb_path)
        self.data: dict[str, Any] = {}
        self.matrix: dict[str, dict[str, Any]] = {}
        self.by_model: dict[str, dict[str, dict[str, Any]]] = {}
        self.load()

    def load(self) -> bool:
        """Load knowledge base from JSON if present."""
        if not self.kb_path.exists():
            return False
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.matrix = self.data.get("matrix", {})
            self.by_model = self.data.get("by_model", {})
            return True
        except Exception:
            return False

    def get_top_strategies(
        self,
        defense_type: str,
        victim_model: Optional[str] = None,
        top_k: int = 3,
        min_attempts: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-K strategies by empirical success rate for a defense type."""
        candidates = []

        # Model-specific data first if available
        model_key = victim_model or "unknown"
        model_matrix = self.by_model.get(model_key, {}).get(defense_type, {})
        source = model_matrix if model_matrix else self.matrix.get(defense_type, {})

        for strategy, stats in source.items():
            if isinstance(stats, dict):
                attempts = stats.get("total_attempts", stats.get("attempts", 0))
                successes = stats.get("total_successes", stats.get("successes", 0))
                rate = stats.get("success_rate", (successes / attempts) if attempts > 0 else 0.0)
            else:
                continue

            if attempts >= min_attempts:
                candidates.append({
                    "strategy": strategy,
                    "success_rate": float(rate),
                    "attempts": int(attempts),
                    "successes": int(successes),
                })

        candidates.sort(key=lambda x: (x["success_rate"], x["attempts"]), reverse=True)
        return candidates[:top_k]
