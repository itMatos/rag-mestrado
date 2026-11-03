"""Rota de busca: só a recuperação, sem LLM."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import embedder_dep, store_dep
from app.api.schemas import SearchRequest, SearchResponse, SourceOut
from app.config import get_settings
from app.embeddings import Embedder
from app.rag.retrieval import retrieve
from app.store import VectorStore

router = APIRouter(tags=["busca e chat"])


@router.post("/search", response_model=SearchResponse, summary="Top-k chunks para uma pergunta (sem LLM)")
def search(
    request: SearchRequest,
    embedder: Annotated[Embedder, Depends(embedder_dep)],
    store: Annotated[VectorStore, Depends(store_dep)],
) -> SearchResponse:
    """Útil para testar a recuperação isoladamente: se o trecho que responde
    a pergunta não aparece aqui, o /chat também não vai conseguir responder."""
    top_k = request.top_k or get_settings().top_k
    results = retrieve(request.question, top_k, request.doc_ids, embedder=embedder, store=store)
    return SearchResponse(
        question=request.question,
        top_k=top_k,
        results=[SourceOut.from_result(r) for r in results],
    )
