"""Geração: recuperação + prompt + LLM. O pipeline RAG completo.

Dividido em duas funções de propósito: `prepare_answer` faz tudo que é
determinístico e barato (busca e montagem do prompt) e `stream_answer` faz a
chamada lenta ao LLM. Assim dá para inspecionar o prompt sem gastar uma
chamada (`scripts/ask.py --dry-run`), e a API pode enviar as fontes ao cliente
mesmo que o LLM falhe no meio do caminho.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from app.embeddings import Embedder
from app.llm import LLMProvider
from app.rag.prompt import NOT_FOUND_ANSWER, Prompt, build_prompt
from app.rag.retrieval import retrieve
from app.store import SearchResult, VectorStore


@dataclass(frozen=True)
class PreparedAnswer:
    question: str
    sources: list[SearchResult]
    prompt: Prompt


def prepare_answer(
    question: str,
    top_k: int | None = None,
    doc_ids: list[str] | None = None,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> PreparedAnswer:
    sources = retrieve(question, top_k, doc_ids, embedder=embedder, store=store)
    return PreparedAnswer(question=question, sources=sources, prompt=build_prompt(question, sources))


def stream_answer(prepared: PreparedAnswer, llm: LLMProvider) -> Iterator[str]:
    # Sem nenhum trecho (índice vazio ou filtro sem resultado) não há o que
    # perguntar ao LLM: ele só poderia responder com conhecimento próprio.
    if not prepared.sources:
        yield NOT_FOUND_ANSWER
        return
    yield from llm.stream(prepared.prompt.system, prepared.prompt.user)
