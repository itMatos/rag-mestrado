"""Dependências injetadas nas rotas (via `Depends`).

Usar `Depends` em vez de chamar as funções diretamente permite trocar as
implementações em testes (`app.dependency_overrides`) sem mexer nas rotas.
"""

from functools import lru_cache

from app.embeddings import Embedder, get_embedder
from app.llm import LLMProvider, create_llm
from app.store import VectorStore, get_vector_store


def embedder_dep() -> Embedder:
    return get_embedder()


def store_dep() -> VectorStore:
    return get_vector_store()


@lru_cache
def llm_dep() -> LLMProvider:
    return create_llm()
