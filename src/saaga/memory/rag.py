"""
SAAGA Defense RAG Layer
=======================
FAISS-backed vector retrieval for past successful defense-breaking exemplars.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


class DefenseRetriever:
    """FAISS-based RAG Retriever for retrieving similar past successful attacks."""

    def __init__(
        self,
        index_path: str | Path = "data/rag/success_defenses.index",
        meta_path: str | Path = "data/rag/success_metadata.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model_name = model_name
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self.model = None
        self.enabled = False

        self._initialize()

    def _initialize(self):
        if not (self.index_path.exists() and self.meta_path.exists()):
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.model = SentenceTransformer(self.model_name)
            self.enabled = True
        except Exception as exc:
            self.enabled = False

    def retrieve(
        self,
        defense_text: str,
        defense_type: Optional[str] = None,
        top_k: int = 20,
        final_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve nearest exemplar attacks for the given defense prompt."""
        if not self.enabled or self.index is None or self.model is None:
            return []

        try:
            import faiss
            import numpy as np

            emb = self.model.encode([defense_text], convert_to_numpy=True)
            faiss.normalize_L2(emb)

            distances, indices = self.index.search(emb, top_k)

            primary = []
            for i in range(min(top_k, len(indices[0]))):
                idx = indices[0][i]
                if idx == -1 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]

                if defense_type and defense_type != "unknown" and meta.get("defense_type") != defense_type:
                    continue

                primary.append({
                    "strategy": meta.get("strategy", "unknown"),
                    "attack": meta.get("attack", ""),
                    "defense_type": meta.get("defense_type", "unknown"),
                    "victim_model": meta.get("victim_model", "unknown"),
                    "distance": float(distances[0][i]),
                })
                if len(primary) >= final_k:
                    break

            results = list(primary)
            if len(results) < final_k:
                for i in range(min(top_k, len(indices[0]))):
                    idx = indices[0][i]
                    if idx == -1 or idx >= len(self.metadata):
                        continue
                    meta = self.metadata[idx]
                    if meta.get("defense_type") == defense_type:
                        continue

                    results.append({
                        "strategy": meta.get("strategy", "unknown"),
                        "attack": meta.get("attack", ""),
                        "defense_type": meta.get("defense_type", "unknown"),
                        "victim_model": meta.get("victim_model", "unknown"),
                        "distance": float(distances[0][i]),
                    })
                    if len(results) >= final_k:
                        break

            return results
        except Exception:
            return []
