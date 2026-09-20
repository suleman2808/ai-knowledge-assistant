"""Local embedding model.

Embeddings are computed on this machine with `sentence-transformers`, not
through an API. For a corpus of a few hundred chunks this is strictly
better: no key, no per-call cost, no rate limit, and ingestion can be rerun
as often as we like while tuning the chunking strategy. It also means the
retrieval half of the system works with no internet connection at all.

The trade-off is a one-off model download (~90 MB) and a few seconds of CPU
on first load, which is why the model is a lazily built singleton rather
than being constructed per call.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_model: Any = None


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or used."""


def get_model() -> Any:
    """Load the sentence-transformer model once and reuse it.

    The first call downloads the weights if they are not cached and takes a
    few seconds. Subsequent calls are instant.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise EmbeddingError(
            f"sentence-transformers is not installed: {exc}"
        ) from exc

    logger.info("Loading embedding model %s", settings.embedding_model)
    try:
        _model = SentenceTransformer(settings.embedding_model)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model {settings.embedding_model!r}: {exc}"
        ) from exc
    return _model


def embed_documents(texts: list[str], *, show_progress: bool = False) -> list[list[float]]:
    """Embed a batch of document chunks for storage.

    Vectors are L2-normalised, which makes cosine similarity equivalent to
    a dot product and keeps scores in a predictable 0-1 range. The
    retrieval layer's score threshold depends on that.
    """
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single user query.

    Kept separate from `embed_documents` even though the implementation is
    currently identical. Asymmetric models (E5, BGE, GTE) require different
    prefixes for queries and passages, so if the model is ever swapped, the
    distinction is already in place at every call site.
    """
    model = get_model()
    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def embedding_dimension() -> int:
    """Report the model's output dimension, for diagnostics.

    The accessor was renamed in sentence-transformers 6; both spellings are
    tried so the project works across versions.
    """
    model = get_model()
    for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        accessor = getattr(model, attribute, None)
        if callable(accessor):
            return int(accessor())
    # Last resort: embed something trivial and measure the result.
    return len(embed_query("dimension probe"))
