"""Recuperação: pergunta -> embedding -> top-k chunks mais similares.

É o "R" do RAG. A qualidade final da resposta depende mais desta etapa do que
do LLM: se o trecho certo não estiver entre os top-k, o LLM não tem como
responder (e, se o prompt estiver bem feito, dirá que não encontrou).
"""

from app.config import get_settings
from app.embeddings import Embedder, get_embedder
from app.store import SearchResult, VectorStore, get_vector_store


def retrieve(
    question: str,
    top_k: int | None = None,
    doc_ids: list[str] | None = None,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> list[SearchResult]:
    embedder = embedder or get_embedder()
    store = store or get_vector_store()
    # A pergunta passa pelo MESMO modelo usado na indexação, mas com o prefixo
    # de consulta ("query: " no e5) — veja app/embeddings/embedder.py.
    query_vector = embedder.embed_query(question)
    return store.search(query_vector, top_k or get_settings().top_k, doc_ids)
