"""Aplicação FastAPI.

Para rodar (de dentro de `backend/`, com o venv ativo):

    uvicorn app.main:app --reload

Swagger em http://localhost:8000/docs
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, search
from app.api.dependencies import llm_dep, store_dep
from app.api.schemas import HealthResponse
from app.config import get_settings
from app.embeddings import get_embedder
from app.llm import LLMProvider
from app.store import VectorStore, get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# Os clientes HTTP logam cada requisição ao Hugging Face/Ollama em INFO; é ruído.
for noisy in ("httpx", "httpx2", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Carrega o modelo de embeddings e abre o ChromaDB na INICIALIZAÇÃO, e não
    # na primeira requisição: (1) a primeira busca não fica ~5 s mais lenta, e
    # (2) um erro de configuração (ex.: modelo trocado sem reindexar) derruba
    # o servidor logo de cara, com a mensagem explicando o que fazer.
    settings = get_settings()
    logger.info("Carregando modelo de embeddings '%s'...", settings.embedding_model)
    get_embedder()
    store = get_vector_store()
    logger.info("Pronto: %d chunks indexados. LLM: %s", store.count(), settings.llm_provider)
    yield


app = FastAPI(
    title="RAG de artigos científicos",
    description="Indexa PDFs de artigos e responde perguntas com base neles, citando as fontes.",
    version="0.1.0",
    lifespan=lifespan,
)

# Sem CORS, o navegador bloqueia chamadas do frontend (localhost:4200) para a
# API (localhost:8000), porque são "origens" diferentes (a porta conta).
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/health", response_model=HealthResponse, tags=["saúde"], summary="Estado do backend")
def health(
    store: Annotated[VectorStore, Depends(store_dep)],
    llm: Annotated[LLMProvider, Depends(llm_dep)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=get_settings().embedding_model,
        llm_provider=llm.name,
        llm_model=llm.model,
        documents=len(store.list_documents()),
        chunks=store.count(),
    )
