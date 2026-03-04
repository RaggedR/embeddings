"""OpenAI embedding model wrappers."""

import os

import numpy as np
from openai import OpenAI

from embench.models.base import EmbeddingModel


class OpenAIEmbed(EmbeddingModel):
    """Wrapper for OpenAI embedding models."""

    def __init__(self, model_id: str, model_name: str, dimensions: int):
        self._model_id = model_id
        self._name = model_name
        self._dim = dimensions
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], batch_size: int = 2048) -> np.ndarray:
        """Embed texts via OpenAI API. Uses batch_size up to 2048 (API limit)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self._model_id,
                input=batch,
            )
            batch_embs = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embs)
        arr = np.array(all_embeddings, dtype=np.float32)
        return self.l2_normalize(arr)


class OpenAISmall(OpenAIEmbed):
    """text-embedding-3-small (1536-dim)."""

    def __init__(self):
        super().__init__("text-embedding-3-small", "openai-small", 1536)


class OpenAILarge(OpenAIEmbed):
    """text-embedding-3-large (3072-dim)."""

    def __init__(self):
        super().__init__("text-embedding-3-large", "openai-large", 3072)
