"""Etapa 2b: armazenamento e busca de vetores (ChromaDB)."""

from functools import lru_cache

from app.config import get_settings
from app.store.vector_store import (
    DocumentInfo,
    EmbeddingModelMismatchError,
    SearchResult,
    VectorStore,
)


@lru_cache
def get_vector_store() -> VectorStore:
    """Store da coleção configurada em `.env` (uma instância por processo)."""
    settings = get_settings()
    return VectorStore(settings.chroma_dir, settings.collection_name, settings.embedding_model)


__all__ = [
    "DocumentInfo",
    "EmbeddingModelMismatchError",
    "SearchResult",
    "VectorStore",
    "get_vector_store",
]
