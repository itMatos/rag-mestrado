"""Busca semântica pura (sem LLM): mostra os top-k chunks e seus scores.

Uso (de dentro de `backend/`, com o venv ativo):

    python3 -m scripts.search "quais métricas de qualidade de código foram usadas?"
    python3 -m scripts.search "technical debt" --top-k 10
    python3 -m scripts.search "..." --doc <doc_id> --full

É a melhor ferramenta para entender o RAG: se o trecho que responde a sua
pergunta não aparece aqui, nenhum LLM vai conseguir responder bem.
"""

import argparse
import sys
import textwrap

from app.config import get_settings
from app.rag.retrieval import retrieve
from app.store import EmbeddingModelMismatchError

PREVIEW_CHARS = 400


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Mostra os top-k chunks para uma pergunta.")
    parser.add_argument("question", help="pergunta (entre aspas)")
    parser.add_argument("--top-k", type=int, default=settings.top_k,
                        help=f"quantos chunks mostrar (padrão: {settings.top_k})")
    parser.add_argument("--doc", action="append", dest="doc_ids", metavar="DOC_ID",
                        help="restringir a um documento (pode repetir)")
    parser.add_argument("--full", action="store_true", help="mostrar o texto completo")
    args = parser.parse_args()

    try:
        results = retrieve(args.question, args.top_k, args.doc_ids)
    except EmbeddingModelMismatchError as exc:
        sys.exit(f"Erro: {exc}")

    if not results:
        if args.doc_ids:
            sys.exit("Nenhum resultado para esses doc_ids. Confira com: python3 -m scripts.index_pdfs --list")
        sys.exit("Nenhum resultado. O índice está vazio? Rode scripts.index_pdfs primeiro.")

    print(f'\nPergunta: "{args.question}"   (top-{args.top_k})\n')
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        pages = (f"p. {chunk.page_start}" if chunk.page_start == chunk.page_end
                 else f"p. {chunk.page_start}–{chunk.page_end}")
        bar = "█" * round(result.score * 20)
        print(f"#{rank}  score {result.score:.4f} {bar}")
        print(f"    [{chunk.title}, {pages}]  chunk {chunk.position}  ({chunk.chunk_id})")
        text = chunk.text if args.full else textwrap.shorten(chunk.text, PREVIEW_CHARS, placeholder=" [...]")
        print(textwrap.indent(textwrap.fill(text, 100), "    "))
        print()


if __name__ == "__main__":
    main()
