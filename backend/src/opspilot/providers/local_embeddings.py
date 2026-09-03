"""Local sentence-transformers embedding provider.

Requires the ``ml`` optional dependency group (``uv sync --extra ml``). The
model is loaded lazily on first use and cached for the process lifetime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opspilot.providers.base import EmbeddingResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class LocalEmbeddingProvider:
    name = "local"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        self.model = model
        self.dim = dim
        self._st: SentenceTransformer | None = None

    def _model(self) -> SentenceTransformer:
        if self._st is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised via message
                raise RuntimeError(
                    "LocalEmbeddingProvider needs the 'ml' extra: run `uv sync --extra ml` "
                    "or set OPSPILOT_EMBEDDING_PROVIDER=fake."
                ) from exc
            self._st = SentenceTransformer(self.model)
            loaded_dim = self._st.get_sentence_embedding_dimension()
            if loaded_dim != self.dim:
                raise ValueError(
                    f"Model {self.model} produces {loaded_dim}-dim vectors but "
                    f"OPSPILOT_EMBEDDING_DIM={self.dim}. Update the setting and migration."
                )
        return self._st

    def embed(self, texts: list[str]) -> EmbeddingResult:
        model = self._model()
        arr = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = [[float(x) for x in row] for row in arr]
        approx_tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResult(vectors=vectors, model=self.model, total_tokens=approx_tokens)
