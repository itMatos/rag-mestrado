"""Pergunta -> resposta com citações, pela linha de comando (RAG completo).

Uso (de dentro de `backend/`, com o venv ativo):

    python3 -m scripts.ask "Qual a taxa de sobrevivência dos problemas introduzidos por IA?"
    python3 -m scripts.ask "..." --provider anthropic
    python3 -m scripts.ask "..." --top-k 8 --doc <doc_id>
    python3 -m scripts.ask "..." --show-prompt     # imprime o prompt antes da resposta
    python3 -m scripts.ask "..." --dry-run         # só o prompt, sem chamar o LLM

A resposta aparece em streaming; no fim são listados os trechos usados como
contexto, com score.
"""

import argparse
import sys
import time

from app.config import get_settings
from app.llm import LLMError, create_llm
from app.rag.generation import prepare_answer, stream_answer
from app.rag.prompt import citation_label
from app.store import EmbeddingModelMismatchError


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Responde uma pergunta com base nos artigos.")
    parser.add_argument("question", help="pergunta (entre aspas)")
    parser.add_argument("--top-k", type=int, default=settings.top_k,
                        help=f"trechos enviados ao LLM (padrão: {settings.top_k})")
    parser.add_argument("--doc", action="append", dest="doc_ids", metavar="DOC_ID",
                        help="restringir a um documento (pode repetir)")
    parser.add_argument("--provider", choices=["ollama", "anthropic"],
                        help=f"sobrescreve LLM_PROVIDER (atual: {settings.llm_provider})")
    parser.add_argument("--show-prompt", action="store_true", help="imprimir o prompt enviado")
    parser.add_argument("--dry-run", action="store_true", help="só montar o prompt, sem chamar o LLM")
    args = parser.parse_args()

    try:
        prepared = prepare_answer(args.question, args.top_k, args.doc_ids)
    except EmbeddingModelMismatchError as exc:
        sys.exit(f"Erro: {exc}")

    if args.show_prompt or args.dry_run:
        rule("PROMPT DE SISTEMA")
        print(prepared.prompt.system)
        rule("MENSAGEM DO USUÁRIO")
        print(prepared.prompt.user)
        if args.dry_run:
            print_sources(prepared.sources)
            return

    try:
        llm = create_llm(args.provider)
        rule(f"RESPOSTA ({llm.name} / {llm.model})")
        started = time.perf_counter()
        first_token_at: float | None = None
        for piece in stream_answer(prepared, llm):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(piece, end="", flush=True)
        print()
    except LLMError as exc:
        sys.exit(f"\nErro do LLM: {exc}")

    total = time.perf_counter() - started
    ttft = (first_token_at or started) - started
    print(f"\n(primeiro pedaço em {ttft:.1f}s, total {total:.1f}s)")
    print_sources(prepared.sources)


def print_sources(sources: list) -> None:
    rule(f"TRECHOS USADOS COMO CONTEXTO ({len(sources)})")
    for i, result in enumerate(sources, start=1):
        print(f"  {i}. score {result.score:.4f}  {citation_label(result.chunk)}  ({result.chunk.chunk_id})")


def rule(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * max(0, 60 - len(title))}")


if __name__ == "__main__":
    main()
