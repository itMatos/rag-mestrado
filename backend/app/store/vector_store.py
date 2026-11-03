"""Banco vetorial: ChromaDB em modo persistente local.

O ChromaDB guarda, para cada chunk: id, vetor, texto e metadados. A busca
encontra os vetores mais próximos do vetor da pergunta usando um índice HNSW
(um grafo que permite busca aproximada rápida sem comparar com todos os vetores).

Decisões:
- NÓS geramos os embeddings (app/embeddings) e passamos os vetores prontos.
  O Chroma tem um embedding padrão embutido, mas usá-lo esconderia qual
  modelo e quais prefixos estão sendo aplicados — justamente o que queremos ver.
- Distância de COSSENO. Com vetores normalizados, distância = 1 - similaridade;
  convertemos para `score = 1 - distância`, que é mais intuitivo (1 = idêntico).
- O nome do modelo de embeddings fica nos metadados da coleção. Se você trocar
  EMBEDDING_MODEL e tentar usar a coleção antiga, o erro é explícito em vez de
  a busca devolver resultados sem sentido.
- Não há banco relacional: os dados do documento (título, nº de páginas) vão
  repetidos nos metadados de cada chunk, e a lista de documentos é obtida
  agrupando os chunks. Para uma biblioteca pessoal (milhares de chunks) é
  rápido o bastante e mantém uma única fonte de verdade.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.ingestion.models import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    # Similaridade de cosseno entre pergunta e chunk. Em modelos como o e5 os
    # valores ficam "comprimidos" (ex.: 0,75–0,90), então compare scores entre
    # si, não com um limiar absoluto.
    score: float


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    title: str
    filename: str
    num_pages: int
    num_chunks: int


class EmbeddingModelMismatchError(Exception):
    """A coleção foi criada com outro modelo de embeddings."""


class VectorStore:
    def __init__(self, path: Path, collection_name: str, embedding_model: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedding_model},
            embedding_function=None,  # os vetores vêm sempre de fora
        )
        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise EmbeddingModelMismatchError(
                f"A coleção '{collection_name}' foi indexada com '{stored_model}', mas "
                f"EMBEDDING_MODEL agora é '{embedding_model}'. Vetores de modelos "
                "diferentes não são comparáveis: use outro COLLECTION_NAME ou apague "
                f"'{path}' e reindexe."
            )

    # ------------------------------------------------------------------ #
    # Escrita
    # ------------------------------------------------------------------ #

    def add_document(self, chunks: list[Chunk], embeddings: np.ndarray, num_pages: int) -> None:
        """Grava os chunks de UM documento, substituindo uma versão anterior.

        Apagar antes de inserir torna a operação idempotente: reindexar o mesmo
        PDF com outro CHUNK_SIZE não deixa chunks antigos misturados.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Número de chunks e de embeddings diferente")
        self.delete_document(chunks[0].doc_id)
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{**c.metadata(), "num_pages": num_pages} for c in chunks],
        )

    def delete_document(self, doc_id: str) -> int:
        """Remove todos os chunks do documento. Devolve quantos foram removidos."""
        ids = self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------ #
    # Leitura
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        return self._collection.count()

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        metadatas = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])["metadatas"]
        docs = _group_documents(metadatas or [])
        return docs[0] if docs else None

    def list_documents(self) -> list[DocumentInfo]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return _group_documents(metadatas)

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        """Chunks de um documento, na ordem em que aparecem no texto."""
        result = self._collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        chunks = [
            _to_chunk(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"] or [], result["metadatas"] or []
            )
        ]
        return sorted(chunks, key=lambda c: c.position)

    def search(
        self, query_embedding: np.ndarray, top_k: int, doc_ids: list[str] | None = None
    ) -> list[SearchResult]:
        """Os `top_k` chunks mais similares ao vetor da consulta.

        `doc_ids` restringe a busca a alguns documentos (filtro por metadado,
        aplicado pelo próprio Chroma antes do ranking).
        """
        if self.count() == 0:
            return []
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        # `query` aceita várias consultas de uma vez; usamos só a primeira ([0]).
        return [
            SearchResult(chunk=_to_chunk(chunk_id, text, meta), score=1.0 - float(distance))
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                (result["documents"] or [[]])[0],
                (result["metadatas"] or [[]])[0],
                (result["distances"] or [[]])[0],
            )
        ]


def _group_documents(metadatas: list) -> list[DocumentInfo]:
    """Agrupa metadados de chunks em um `DocumentInfo` por documento."""
    docs: dict[str, DocumentInfo] = {}
    for meta in metadatas:
        doc_id = str(meta["doc_id"])
        current = docs.get(doc_id)
        docs[doc_id] = DocumentInfo(
            doc_id=doc_id,
            title=str(meta["title"]),
            filename=str(meta["filename"]),
            num_pages=int(meta["num_pages"]),
            num_chunks=(current.num_chunks if current else 0) + 1,
        )
    return sorted(docs.values(), key=lambda d: d.title.lower())


def _to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        filename=str(meta["filename"]),
        page_start=int(meta["page_start"]),
        page_end=int(meta["page_end"]),
        position=int(meta["position"]),
        text=text,
    )
