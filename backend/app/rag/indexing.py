"""Indexação: PDF -> extração -> chunks -> embeddings -> ChromaDB.

Junta as etapas anteriores num único fluxo. Tudo que é "configurável" pode ser
passado como argumento (em vez de vir só do `.env`) para que o script de
avaliação consiga indexar o mesmo PDF com parâmetros diferentes.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.embeddings import Embedder, get_embedder
from app.ingestion import chunk_document, extract_pdf_bytes
from app.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingReport:
    doc_id: str
    title: str
    filename: str
    num_pages: int
    num_chunks: int
    references_page: int | None
    # True se o documento já existia e foi substituído.
    replaced: bool
    # Chunks com mais tokens do que o modelo aceita. O excesso é cortado no
    # embedding, ou seja, o final desses chunks fica invisível para a busca.
    # Se for > 0, diminua CHUNK_SIZE.
    truncated_chunks: int
    max_chunk_tokens: int
    seconds: float


def index_pdf(
    path: Path,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    remove_references: bool | None = None,
) -> IndexingReport:
    """Atalho para indexar um PDF do disco (usado pelos scripts)."""
    return index_pdf_bytes(
        path.read_bytes(),
        path.name,
        embedder=embedder,
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        remove_references=remove_references,
    )


def index_pdf_bytes(
    data: bytes,
    filename: str,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    remove_references: bool | None = None,
) -> IndexingReport:
    settings = get_settings()
    embedder = embedder or get_embedder()
    store = store or get_vector_store()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if remove_references is None:
        remove_references = settings.remove_references

    started = time.perf_counter()
    doc = extract_pdf_bytes(data, filename, remove_references=remove_references)
    chunks = chunk_document(doc, chunk_size, chunk_overlap)

    token_counts = [embedder.count_tokens(c.text) for c in chunks]
    truncated = sum(1 for n in token_counts if n > embedder.max_tokens)
    if truncated:
        logger.warning(
            "%s: %d chunks passam de %d tokens e serão truncados; reduza CHUNK_SIZE.",
            filename, truncated, embedder.max_tokens,
        )

    replaced = store.has_document(doc.doc_id)
    embeddings = embedder.embed_passages([c.text for c in chunks])
    store.add_document(chunks, embeddings, num_pages=doc.num_pages)

    return IndexingReport(
        doc_id=doc.doc_id,
        title=doc.title,
        filename=doc.filename,
        num_pages=doc.num_pages,
        num_chunks=len(chunks),
        references_page=doc.references_page,
        replaced=replaced,
        truncated_chunks=truncated,
        max_chunk_tokens=max(token_counts, default=0),
        seconds=time.perf_counter() - started,
    )
