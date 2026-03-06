"""Scientific document embedding models (SPECTER2, SciNCL)."""

import numpy as np
import torch

from embench.models.base import EmbeddingModel


class SentenceTransformerEmbed(EmbeddingModel):
    """Wrapper for any sentence-transformers model."""

    def __init__(self, model_id: str, model_name: str, dimensions: int):
        from sentence_transformers import SentenceTransformer

        self._name = model_name
        self._dim = dimensions
        self._model = SentenceTransformer(model_id)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        arr = np.array(embs, dtype=np.float32)
        return self.l2_normalize(arr)


class Specter2(EmbeddingModel):
    """SPECTER2 with proximity adapter (768-dim).

    Uses the `adapters` library to load the base model + task adapter.
    CLS token pooling as specified by the model authors.
    """

    def __init__(self):
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self._model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self._model.load_adapter(
            "allenai/specter2",
            source="hf",
            load_as="specter2_proximity",
            set_active=True,
        )
        self._model.eval()

    @property
    def name(self) -> str:
        return "specter2"

    @property
    def dim(self) -> int:
        return 768

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            with torch.no_grad():
                output = self._model(**inputs)
            # CLS token embedding
            cls_embs = output.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(cls_embs)
            if (i // batch_size) % 10 == 0:
                print(f"  SPECTER2: {min(i + batch_size, len(texts))}/{len(texts)}")
        arr = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return self.l2_normalize(arr)


class SciNCL(SentenceTransformerEmbed):
    """SciNCL — SciBERT + citation graph contrastive learning (768-dim)."""

    def __init__(self):
        super().__init__(
            "malteos/scincl",
            "scincl",
            768,
        )
