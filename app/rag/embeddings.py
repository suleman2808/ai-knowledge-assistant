"""Local embedding model, with two interchangeable runtimes.

Embeddings are computed on this machine, never through an API. For a
corpus of a few hundred chunks that is strictly better: no key, no
per-call cost, no rate limit, and ingestion can be rerun freely while
tuning the chunking strategy. Retrieval works with no internet at all.

## Two backends, one model

Both run `all-MiniLM-L6-v2` and produce the same 384-dimension vectors:

- **onnx** (default) — the model's ONNX export on `onnxruntime`. No
  PyTorch, so it installs in seconds rather than pulling ~1.5 GB, loads
  faster, and runs on machines where PyTorch cannot.
- **sentence-transformers** — the reference implementation on PyTorch.
  Supports any Hugging Face embedding model, not just this one.

The ONNX default was not the original plan. It was adopted when Windows
Smart App Control began refusing PyTorch's unsigned `c10.dll`
(`WinError 4551`) on the development machine, which broke every
inquiry. Asking every user to weaken an OS security feature is not an
acceptable install step, so the runtime changed rather than the machine.
Anyone cloning onto a similarly locked-down Windows box would have hit
the same wall.

Switch with `EMBEDDING_BACKEND=sentence-transformers` in `.env`, then
re-run ingestion: vectors from different runtimes are near-identical but
should never be mixed in one index.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# The only model Chroma ships as an ONNX export.
ONNX_MODEL = "all-MiniLM-L6-v2"

_backend: "_Backend | None" = None


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or used."""


class _Backend:
    """Common shape for the two runtimes."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class _OnnxBackend(_Backend):
    name = "onnx"

    def __init__(self) -> None:
        model = settings.embedding_model.split("/")[-1]
        if model != ONNX_MODEL:
            raise EmbeddingError(
                f"The ONNX backend only provides {ONNX_MODEL}, but "
                f"EMBEDDING_MODEL is {settings.embedding_model!r}. Use "
                f"EMBEDDING_BACKEND=sentence-transformers for other models."
            )
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        except Exception as exc:
            raise EmbeddingError(f"ONNX runtime unavailable: {exc}") from exc
        self._fn = ONNXMiniLM_L6_V2()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._fn(texts), dtype=np.float32)


class _SentenceTransformersBackend(_Backend):
    name = "sentence-transformers"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            # Catches OSError as well as ImportError. A DLL refused by the
            # operating system raises OSError at import time, and an
            # ImportError-only handler lets it escape as an "unexpected"
            # failure — which is exactly what happened before this change.
            raise EmbeddingError(
                f"sentence-transformers could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._model = SentenceTransformer(settings.embedding_model)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model {settings.embedding_model!r}: {exc}"
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, batch_size=32, convert_to_numpy=True)


BACKENDS: dict[str, type[_Backend]] = {
    "onnx": _OnnxBackend,
    "sentence-transformers": _SentenceTransformersBackend,
}


def get_backend() -> _Backend:
    """Load the configured runtime once and reuse it."""
    global _backend
    if _backend is not None:
        return _backend

    choice = settings.embedding_backend.lower().strip()
    backend_cls = BACKENDS.get(choice)
    if backend_cls is None:
        raise EmbeddingError(
            f"Unknown EMBEDDING_BACKEND {choice!r}. Choose from: {', '.join(BACKENDS)}"
        )

    logger.info("Loading embedding model %s via %s", settings.embedding_model, choice)
    _backend = backend_cls()
    return _backend


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows.

    Done here, for both backends, rather than trusting each runtime's own
    option. Normalised vectors make cosine similarity a dot product and
    keep scores in a predictable range, and the retrieval threshold was
    calibrated on exactly that range.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _encode(texts: list[str]) -> np.ndarray:
    try:
        vectors = get_backend().encode(texts)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {type(exc).__name__}: {exc}") from exc
    return _normalise(np.atleast_2d(vectors))


def embed_documents(texts: list[str], *, show_progress: bool = False) -> list[list[float]]:
    """Embed a batch of document chunks for storage."""
    if not texts:
        return []
    del show_progress  # batches are small enough that progress adds nothing
    return [v.tolist() for v in _encode(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single user query.

    Kept separate from `embed_documents` even though the implementation is
    currently identical. Asymmetric models (E5, BGE, GTE) require different
    prefixes for queries and passages, so if the model is ever swapped the
    distinction is already in place at every call site.
    """
    return _encode([text])[0].tolist()


def embedding_dimension() -> int:
    """Report the output dimension, for diagnostics."""
    return len(embed_query("dimension probe"))


def backend_name() -> str:
    """Which runtime is in use, for `/health` and ingestion reports."""
    return get_backend().name
