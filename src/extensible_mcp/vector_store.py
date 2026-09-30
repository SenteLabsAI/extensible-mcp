# Copyright (c) 2026 Matthew Fuchs
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from .types import SearchResult, ToolRecord

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# fastembed pads every text in a batch to the batch's longest (up to the
# model's 256 tokens), so the attention buffers grow with batch size x 256^2.
# Indexing ~120 tool descriptions as one batch peaks above 1 GB of RSS;
# 16 at a time keeps the index build to a few hundred MB.
_EMBED_BATCH_SIZE = 16


class VectorStore:
    def __init__(
        self, model_name: str = _DEFAULT_MODEL, embed_batch_size: int = _EMBED_BATCH_SIZE
    ) -> None:
        if embed_batch_size < 1:
            raise ValueError(f"embed_batch_size must be positive, got {embed_batch_size}")
        self._model = TextEmbedding(model_name=model_name)
        self._embed_batch_size = embed_batch_size
        self._tools: list[ToolRecord] = []
        self._embeddings: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = np.array(list(self._model.embed(texts, batch_size=self._embed_batch_size)), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def index(self, tools: list[ToolRecord]) -> None:
        if not tools:
            self._tools = []
            self._embeddings = None
            return
        self._tools = list(tools)
        texts = [t.embedding_text for t in self._tools]
        self._embeddings = self._encode(texts)

    def add(self, tools: list[ToolRecord]) -> None:
        """Add tools incrementally to the existing index."""
        if not tools:
            return
        texts = [t.embedding_text for t in tools]
        new_embeddings = self._encode(texts)
        self._tools.extend(tools)
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self._embeddings is None or len(self._tools) == 0:
            return []
        query_vec = self._encode([query])[0]
        scores = self._embeddings @ query_vec
        k = min(top_k, len(self._tools))
        if k >= len(self._tools):
            top_indices = np.argsort(-scores)[:k]
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [
            SearchResult(tool=self._tools[i], score=float(scores[i]))
            for i in top_indices
        ]
