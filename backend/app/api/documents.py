"""Rotas de documentos: upload/indexação, listagem, remoção e inspeção de chunks."""

import logging
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import embedder_dep, store_dep
from app.api.schemas import (
    ChunkOut,
    DeleteResponse,
    DocumentChunksResponse,
    DocumentListResponse,
    DocumentOut,
    UploadResponse,
    UploadResult,
)
from app.config import get_settings
from app.embeddings import Embedder
from app.ingestion import PdfExtractionError
from app.rag.indexing import index_pdf_bytes
from app.store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documentos"])

# Indexar usa bastante CPU/memória (modelo de embeddings) e faz "apaga e
# reinsere" no Chroma. Serializar as indexações evita que dois uploads
# simultâneos do mesmo PDF se atropelem. Buscas não passam por este lock.
_index_lock = threading.Lock()


@router.post("", response_model=UploadResponse, summary="Envia e indexa um ou mais PDFs")
def upload_documents(
    files: Annotated[list[UploadFile], File(description="Um ou mais arquivos PDF.")],
    embedder: Annotated[Embedder, Depends(embedder_dep)],
    store: Annotated[VectorStore, Depends(store_dep)],
) -> UploadResponse:
    """Cada arquivo é processado de forma independente: um PDF com problema
    não impede a indexação dos outros. O resultado de cada um vem em `results`.

    A rota é `def` (e não `async def`) de propósito: extração e embeddings são
    trabalho de CPU bloqueante, e o FastAPI executa rotas `def` numa thread
    separada, sem travar as outras requisições.
    """
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    results: list[UploadResult] = []
    for upload in files:
        filename = upload.filename or "sem_nome.pdf"
        data = upload.file.read(max_bytes + 1)

        error = _validate_pdf(data, max_bytes)
        if error:
            results.append(UploadResult(filename=filename, status="error", error=error))
            continue

        try:
            with _index_lock:
                report = index_pdf_bytes(data, filename, embedder=embedder, store=store)
        except PdfExtractionError as exc:
            results.append(UploadResult(filename=filename, status="error", error=str(exc)))
            continue

        logger.info("Indexado %s (%s): %d chunks", filename, report.doc_id, report.num_chunks)
        results.append(
            UploadResult(
                filename=filename,
                status="replaced" if report.replaced else "indexed",
                document=DocumentOut(
                    doc_id=report.doc_id,
                    title=report.title,
                    filename=report.filename,
                    num_pages=report.num_pages,
                    num_chunks=report.num_chunks,
                ),
                references_page=report.references_page,
                truncated_chunks=report.truncated_chunks,
            )
        )
    return UploadResponse(results=results)


def _validate_pdf(data: bytes, max_bytes: int) -> str | None:
    if len(data) > max_bytes:
        return f"Arquivo maior que o limite de {max_bytes // (1024 * 1024)} MB."
    # Confere a "assinatura" do arquivo em vez de confiar na extensão ou no
    # content-type, que o cliente pode mandar errado.
    if not data.startswith(b"%PDF-"):
        return "O arquivo não é um PDF."
    return None


@router.get("", response_model=DocumentListResponse, summary="Lista os documentos indexados")
def list_documents(store: Annotated[VectorStore, Depends(store_dep)]) -> DocumentListResponse:
    docs = store.list_documents()
    return DocumentListResponse(
        documents=[DocumentOut.from_info(d) for d in docs],
        total_chunks=sum(d.num_chunks for d in docs),
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Remove um documento e seus chunks",
    responses={404: {"description": "Documento não encontrado"}},
)
def delete_document(doc_id: str, store: Annotated[VectorStore, Depends(store_dep)]) -> DeleteResponse:
    with _index_lock:
        deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado.")
    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)


@router.get(
    "/{doc_id}/chunks",
    response_model=DocumentChunksResponse,
    summary="Mostra os chunks de um documento (para depurar o chunking)",
    responses={404: {"description": "Documento não encontrado"}},
)
def get_document_chunks(
    doc_id: str,
    store: Annotated[VectorStore, Depends(store_dep)],
    embedder: Annotated[Embedder, Depends(embedder_dep)],
) -> DocumentChunksResponse:
    info = store.get_document(doc_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado.")
    chunks = []
    for chunk in store.get_document_chunks(doc_id):
        tokens = embedder.count_tokens(chunk.text)
        chunks.append(
            ChunkOut(
                chunk_id=chunk.chunk_id,
                position=chunk.position,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                word_count=chunk.word_count,
                token_count=tokens,
                truncated=tokens > embedder.max_tokens,
                text=chunk.text,
            )
        )
    return DocumentChunksResponse(document=DocumentOut.from_info(info), chunks=chunks)
