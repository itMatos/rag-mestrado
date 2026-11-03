"""Indexa PDFs no ChromaDB e gerencia o índice pela linha de comando.

Uso (de dentro de `backend/`, com o venv ativo):

    python3 -m scripts.index_pdfs artigo1.pdf artigo2.pdf
    python3 -m scripts.index_pdfs ~/pasta/com/pdfs/          # todos os .pdf da pasta
    python3 -m scripts.index_pdfs --list
    python3 -m scripts.index_pdfs --delete <doc_id>

A primeira execução baixa o modelo de embeddings (~1 GB para o e5-base).
"""

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedder
from app.ingestion import PdfExtractionError
from app.rag.indexing import index_pdf
from app.store import EmbeddingModelMismatchError, VectorStore, get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa PDFs no ChromaDB.")
    parser.add_argument("paths", nargs="*", type=Path, help="arquivos PDF ou pastas")
    parser.add_argument("--list", action="store_true", help="listar documentos indexados")
    parser.add_argument("--delete", metavar="DOC_ID", help="remover um documento do índice")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="AVISO: %(message)s")

    try:
        store = get_vector_store()
    except EmbeddingModelMismatchError as exc:
        sys.exit(f"Erro: {exc}")

    if args.delete:
        removed = store.delete_document(args.delete)
        print(f"{removed} chunks removidos." if removed else "Documento não encontrado.")
    if args.paths:
        index_paths(collect_pdfs(args.paths), store)
    if args.list or not (args.paths or args.delete):
        print_documents(store)


def collect_pdfs(paths: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for path in paths:
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        elif path.is_file():
            pdfs.append(path)
        else:
            print(f"Ignorando (não encontrado): {path}")
    return pdfs


def index_paths(pdfs: list[Path], store: VectorStore) -> None:
    settings = get_settings()
    print(f"Carregando modelo de embeddings '{settings.embedding_model}'...")
    embedder = get_embedder()
    print(f"Modelo pronto: {embedder.dimension} dimensões, máx. {embedder.max_tokens} tokens.")
    print(f"Chunking: {settings.chunk_size} palavras, overlap {settings.chunk_overlap}.\n")

    for pdf in pdfs:
        print(f"→ {pdf.name}")
        try:
            report = index_pdf(pdf, embedder=embedder, store=store)
        except PdfExtractionError as exc:
            print(f"  ERRO: {exc}\n")
            continue
        refs = (f"referências cortadas na p. {report.references_page}"
                if report.references_page else "referências não detectadas")
        print(f"  título:  {report.title}")
        print(f"  doc_id:  {report.doc_id}{'  (substituído)' if report.replaced else ''}")
        print(f"  {report.num_pages} páginas, {report.num_chunks} chunks, {refs}")
        print(f"  maior chunk: {report.max_chunk_tokens} tokens"
              f"{f' — {report.truncated_chunks} TRUNCADOS' if report.truncated_chunks else ''}")
        print(f"  {report.seconds:.1f}s\n")


def print_documents(store: VectorStore) -> None:
    docs = store.list_documents()
    print(f"{len(docs)} documento(s), {store.count()} chunks no índice:\n")
    for doc in docs:
        print(f"  {doc.doc_id}  {doc.num_pages:>3} págs  {doc.num_chunks:>4} chunks  {doc.title}")


if __name__ == "__main__":
    main()
