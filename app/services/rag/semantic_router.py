"""Semantic Router — classifies questions into legal branches without LLM calls.

Uses average embeddings (prototypes) computed from the vector store corpus.
A prototype is the mean embedding of all chunks belonging to a legal branch.
Classification is a single cosine similarity against all prototypes.

Zero hardcodes: prototypes are regenerated whenever the corpus changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.services.rag.embeddings import embedding_service
from app.services.rag.vector_store import legislation_vector_store

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Classifies legal questions by branch using prototype embeddings."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache_path = (
            Path(self.settings.processed_dir)
            / "semantic_router"
            / "branch_prototypes.json"
        )
        self._prototypes: dict[str, np.ndarray] | None = None
        self._cache_version: str = ""

    def _needs_refresh(self) -> bool:
        stats = legislation_vector_store.get_stats()
        current_version = f"{stats.get('total', 0)}:{stats.get('last_update') or ''}"
        return self._prototypes is None or self._cache_version != current_version

    def _build_prototypes(self) -> dict[str, np.ndarray]:
        """Compute mean embedding per branch from the vector store corpus.

        Called automatically when the corpus changes. No manual intervention needed.
        """
        prototypes = legislation_vector_store.get_branch_prototypes()

        # Persist as JSON (numpy arrays serialized as lists)
        serializable: dict[str, list[float]] = {}
        for branch, vec in prototypes.items():
            serializable[branch] = vec.tolist()
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(serializable), encoding="utf-8")
        except Exception as exc:
            logger.warning("SemanticRouter: failed to cache prototypes: %s", exc)

        stats = legislation_vector_store.get_stats()
        self._cache_version = f"{stats.get('total', 0)}:{stats.get('last_update') or ''}:{get_settings().local_embedding_model}"
        self._prototypes = prototypes
        return prototypes

    def _load_or_build(self) -> dict[str, np.ndarray]:
        if not self._needs_refresh():
            return self._prototypes  # type: ignore[return-value]

        # Try loading from cache first
        if self._cache_path.exists():
            try:
                cached = json.loads(self._cache_path.read_text(encoding="utf-8"))
                stats = legislation_vector_store.get_stats()
                current_version = f"{stats.get('total', 0)}:{stats.get('last_update') or ''}:{get_settings().local_embedding_model}"
                if current_version == self._cache_version:
                    prototypes: dict[str, np.ndarray] = {}
                    for branch, vec_list in cached.items():
                        if vec_list:
                            prototypes[branch] = np.array(vec_list, dtype=np.float32)
                    self._prototypes = prototypes
                    self._cache_version = current_version
                    return prototypes
            except Exception as exc:
                logger.warning("SemanticRouter: cache load failed, rebuilding: %s", exc)

        return self._build_prototypes()

    async def classify(self, question: str) -> tuple[str, float]:
        """Classify a question into a legal branch.

        Returns (branch_name, confidence_score).
        confidence: 1.0 = perfect match, 0.0 = random.
        If confidence < 0.4, the result is unreliable — fall back to LLM.
        """
        prototypes = self._load_or_build()
        if not prototypes:
            return "indeterminado", 0.0

        query_vec = np.array(
            await embedding_service.embed_query(question), dtype=np.float32
        )
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0:
            return "indeterminado", 0.0

        best_branch = "indeterminado"
        best_sim = -1.0
        sims: dict[str, float] = {}

        for branch, proto in prototypes.items():
            proto_norm = float(np.linalg.norm(proto))
            if proto_norm == 0:
                continue
            sim = float(np.dot(query_vec, proto)) / (query_norm * proto_norm)
            sims[branch] = sim
            if sim > best_sim:
                best_sim = sim
                best_branch = branch

        # Confidence: difference between best and second-best, scaled
        sorted_sims = sorted(sims.values(), reverse=True)
        if len(sorted_sims) >= 2:
            margin = sorted_sims[0] - sorted_sims[1]
        else:
            margin = best_sim

        # Heuristic: cosine sim > 0.7 is good, margin > 0.1 is decisive
        confidence = min(1.0, max(0.0, (best_sim - 0.3) * 1.5 + margin * 2.0))
        confidence = round(confidence, 4)

        return best_branch, confidence


semantic_router = SemanticRouter()
