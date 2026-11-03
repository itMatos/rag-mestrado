"""Schemas Pydantic de todas as requests e responses da API.

São o contrato com o frontend: o Swagger (/docs) é gerado a partir deles, e
qualquer campo novo aparece lá automaticamente. Ficam separados das
dataclasses internas (app/ingestion/models.py, app/store) para que mudanças
internas não quebrem a API sem querer.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.ingestion.models import Chunk
from app.rag.prompt import citation_label
from app.store import DocumentInfo, SearchResult

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


# ---------------------------------------------------------------------- #
# Documentos
# ---------------------------------------------------------------------- #


class DocumentOut(BaseModel):
    doc_id: str = Field(description="Hash do conteúdo do PDF (16 caracteres hex).")
    title: str
    filename: str
    num_pages: int = Field(description="Total de páginas do arquivo.")
    num_chunks: int

    @classmethod
    def from_info(cls, info: DocumentInfo) -> "DocumentOut":
        return cls(
            doc_id=info.doc_id,
            title=info.title,
            filename=info.filename,
            num_pages=info.num_pages,
            num_chunks=info.num_chunks,
        )


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total_chunks: int


class UploadResult(BaseModel):
    filename: str
    status: Literal["indexed", "replaced", "error"] = Field(
        description="`indexed`: novo; `replaced`: já existia e foi reindexado; `error`: falhou."
    )
    document: DocumentOut | None = Field(default=None, description="Ausente quando status = error.")
    references_page: int | None = Field(
        default=None, description="Página a partir da qual a seção de referências foi removida."
    )
    truncated_chunks: int = Field(
        default=0, description="Chunks maiores que o limite do modelo de embeddings (final ignorado)."
    )
    error: str | None = Field(default=None, description="Presente só quando status = error.")


class UploadResponse(BaseModel):
    results: list[UploadResult] = Field(description="Um item por arquivo, na ordem do envio.")


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int


class ChunkOut(BaseModel):
    chunk_id: str
    position: int
    page_start: int
    page_end: int
    word_count: int
    token_count: int = Field(description="Tokens que o modelo de embeddings vê (com prefixo).")
    truncated: bool = Field(description="True se token_count passa do limite do modelo.")
    text: str


class DocumentChunksResponse(BaseModel):
    document: DocumentOut
    chunks: list[ChunkOut]


# ---------------------------------------------------------------------- #
# Busca e chat
# ---------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    question: Question = Field(examples=["Qual porcentagem dos problemas introduzidos por IA ainda sobrevive?"])
    top_k: int | None = Field(default=None, ge=1, le=50, description="Padrão: TOP_K do .env.")
    doc_ids: list[str] | None = Field(
        default=None, description="Restringe a busca a estes documentos. Vazio/ausente = todos."
    )


class SearchRequest(QueryRequest):
    pass


class ChatRequest(QueryRequest):
    pass


class SourceOut(BaseModel):
    """Um chunk recuperado, com tudo que o frontend precisa para exibi-lo."""

    chunk_id: str
    doc_id: str
    title: str
    filename: str
    page_start: int
    page_end: int
    position: int
    citation: str = Field(description="Rótulo no formato usado nas respostas: [Título, p. X].")
    score: float = Field(description="Similaridade de cosseno (maior = mais parecido).")
    text: str

    @classmethod
    def from_result(cls, result: SearchResult) -> "SourceOut":
        chunk: Chunk = result.chunk
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            filename=chunk.filename,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            position=chunk.position,
            citation=citation_label(chunk),
            score=round(result.score, 4),
            text=chunk.text,
        )


class SearchResponse(BaseModel):
    question: str
    top_k: int
    results: list[SourceOut] = Field(description="Ordenados do mais para o menos similar.")


# Payloads dos eventos SSE de POST /chat (campo `data`, em JSON).


class TokenEvent(BaseModel):
    text: str = Field(description="Próximo pedaço da resposta; concatene na ordem.")


class ErrorEvent(BaseModel):
    message: str


class SourcesEvent(BaseModel):
    sources: list[SourceOut]


class DoneEvent(BaseModel):
    provider: str
    model: str
    elapsed_ms: int
    error: bool = Field(description="True se houve um evento `error` antes deste.")


# ---------------------------------------------------------------------- #
# Saúde
# ---------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: Literal["ok"]
    embedding_model: str
    llm_provider: str
    llm_model: str
    documents: int
    chunks: int
