"""Rota de chat: RAG completo com resposta em streaming via Server-Sent Events.

SSE é um protocolo simples sobre HTTP: a resposta fica aberta e o servidor
escreve blocos de texto, cada um no formato

    event: <nome>
    data: <conteúdo>
    <linha em branco>

Sequência de eventos desta rota:

    token  (0..N vezes)  {"text": "..."}         pedaços da resposta, em ordem
    error  (0 ou 1 vez)  {"message": "..."}      falha do LLM no meio do caminho
    sources (1 vez)      {"sources": [...]}      trechos usados como contexto
    done   (1 vez)       {"provider", "model", "elapsed_ms", "error"}

O `data` é sempre JSON. Isso resolve um detalhe do protocolo: uma linha
`data:` não pode conter quebra de linha, e as respostas do LLM têm várias.
Em JSON, a quebra de linha vira a sequência de escape "\\n", numa linha só.

`sources` vem DEPOIS da resposta (como pedido na especificação) e é enviado
mesmo se o LLM falhar, para que dê para ver o que a busca encontrou.
"""

import logging
import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.dependencies import embedder_dep, llm_dep, store_dep
from app.api.schemas import ChatRequest, DoneEvent, ErrorEvent, SourceOut, SourcesEvent, TokenEvent
from app.embeddings import Embedder
from app.llm import LLMError, LLMProvider
from app.rag.generation import PreparedAnswer, prepare_answer, stream_answer
from app.store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["busca e chat"])


@router.post(
    "/chat",
    summary="Responde uma pergunta com base nos artigos (streaming SSE)",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Fluxo SSE com eventos token, error, sources e done.",
            "content": {"text/event-stream": {}},
        }
    },
)
def chat(
    request: ChatRequest,
    embedder: Annotated[Embedder, Depends(embedder_dep)],
    store: Annotated[VectorStore, Depends(store_dep)],
    llm: Annotated[LLMProvider, Depends(llm_dep)],
) -> StreamingResponse:
    # A recuperação acontece ANTES de abrir o stream: se ela falhar, o cliente
    # recebe um erro HTTP normal (com status code), e não um stream quebrado.
    prepared = prepare_answer(
        request.question, request.top_k, request.doc_ids, embedder=embedder, store=store
    )
    return StreamingResponse(
        _chat_events(prepared, llm),
        media_type="text/event-stream",
        headers={
            # Evita que proxies/navegadores guardem ou bufferizem o stream.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _chat_events(prepared: PreparedAnswer, llm: LLMProvider) -> Iterator[str]:
    started = time.perf_counter()
    failed = False
    try:
        for piece in stream_answer(prepared, llm):
            yield _sse("token", TokenEvent(text=piece))
    except LLMError as exc:
        failed = True
        yield _sse("error", ErrorEvent(message=str(exc)))
    except Exception:
        # Depois que o stream começou não dá mais para mudar o status HTTP
        # (já foi enviado 200). O jeito de avisar o cliente é um evento.
        logger.exception("Erro inesperado durante o streaming do chat")
        failed = True
        yield _sse("error", ErrorEvent(message="Erro interno ao gerar a resposta."))

    yield _sse("sources", SourcesEvent(sources=[SourceOut.from_result(r) for r in prepared.sources]))
    yield _sse(
        "done",
        DoneEvent(
            provider=llm.name,
            model=llm.model,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            error=failed,
        ),
    )


def _sse(event: str, payload: BaseModel) -> str:
    return f"event: {event}\ndata: {payload.model_dump_json()}\n\n"
