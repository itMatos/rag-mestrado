"""Mostra o que a etapa de ingestão produz a partir de um PDF.

Uso (de dentro de `backend/`, com o venv ativo):

    python -m scripts.inspect_chunks caminho/artigo.pdf
    python -m scripts.inspect_chunks artigo.pdf --chunk-size 150 --overlap 30
    python -m scripts.inspect_chunks artigo.pdf --pages        # texto limpo por página
    python -m scripts.inspect_chunks artigo.pdf --full         # chunks completos
    python -m scripts.inspect_chunks artigo.pdf --keep-references

Nada é salvo: o script só extrai, divide e imprime.
"""

import argparse
import statistics
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion import Chunk, ExtractedDocument, PdfExtractionError, chunk_document, extract_pdf

PREVIEW_CHARS = 300


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Inspeciona extração e chunking de um PDF.")
    parser.add_argument("pdf", type=Path, help="caminho do arquivo PDF")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size,
                        help=f"palavras por chunk (padrão: {settings.chunk_size})")
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap,
                        help=f"palavras de overlap (padrão: {settings.chunk_overlap})")
    parser.add_argument("--keep-references", action="store_true",
                        help="não remover a seção de referências")
    parser.add_argument("--pages", action="store_true",
                        help="imprimir o texto limpo de cada página antes dos chunks")
    parser.add_argument("--full", action="store_true",
                        help="imprimir o texto completo dos chunks (padrão: prévia)")
    parser.add_argument("--limit", type=int, default=None,
                        help="mostrar só os N primeiros chunks")
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"Arquivo não encontrado: {args.pdf}")

    try:
        doc = extract_pdf(args.pdf, remove_references=not args.keep_references)
        chunks = chunk_document(doc, args.chunk_size, args.overlap)
    except (PdfExtractionError, ValueError) as exc:
        sys.exit(f"Erro: {exc}")

    print_document(doc, args.keep_references)
    if args.pages:
        print_pages(doc)
    print_chunk_stats(chunks, args.chunk_size, args.overlap)
    for chunk in chunks[: args.limit]:
        print_chunk(chunk, args.full)


def print_document(doc: ExtractedDocument, keep_references: bool) -> None:
    rule("DOCUMENTO")
    print(f"Título:      {doc.title}")
    print(f"Arquivo:     {doc.filename}")
    print(f"doc_id:      {doc.doc_id}")
    print(f"Páginas:     {doc.num_pages} no arquivo, {len(doc.pages)} com texto útil")
    if keep_references:
        print("Referências: remoção desligada (--keep-references)")
    elif doc.references_page is not None:
        print(f"Referências: detectadas e removidas a partir da página {doc.references_page}")
    else:
        print("Referências: título da seção NÃO encontrado (nada foi removido)")


def print_pages(doc: ExtractedDocument) -> None:
    for page in doc.pages:
        rule(f"PÁGINA {page.page_number} ({len(page.text.split())} palavras)")
        print(page.text)


def print_chunk_stats(chunks: list[Chunk], chunk_size: int, overlap: int) -> None:
    rule("CHUNKS")
    sizes = [c.word_count for c in chunks]
    print(f"Parâmetros:  chunk_size={chunk_size} palavras, overlap={overlap} palavras "
          f"(passo de {chunk_size - overlap})")
    print(f"Total:       {len(chunks)} chunks")
    if sizes:
        print(f"Palavras:    mín {min(sizes)}, média {statistics.mean(sizes):.0f}, máx {max(sizes)}")
        crossing = sum(1 for c in chunks if c.page_start != c.page_end)
        print(f"Atravessam quebra de página: {crossing}")


def print_chunk(chunk: Chunk, full: bool) -> None:
    pages = (f"p. {chunk.page_start}" if chunk.page_start == chunk.page_end
             else f"p. {chunk.page_start}–{chunk.page_end}")
    print(f"\n--- [{chunk.position}] {chunk.chunk_id} | {pages} | {chunk.word_count} palavras ---")
    text = chunk.text
    if not full and len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + " [...]"
    print(text)


def rule(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * max(0, 60 - len(title))}")


if __name__ == "__main__":
    main()
