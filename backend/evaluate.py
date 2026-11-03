"""Avaliação da RECUPERAÇÃO: recall@k e MRR a partir de um conjunto de perguntas.

Por que avaliar só a recuperação?
Ela é a parte que mais influencia a qualidade final e a mais fácil de medir
objetivamente: dada uma pergunta, o trecho certo apareceu entre os k primeiros?
Avaliar a resposta do LLM exige julgar texto livre (outro LLM ou uma pessoa).
Se a recuperação falha, a resposta não tem salvação; se ela acerta, o resto é
problema de prompt/modelo.

Métricas (calculadas sobre todas as perguntas de eval.json):
- recall@k: fração das perguntas em que PELO MENOS UM dos k primeiros chunks
  vem do documento esperado e cobre uma das páginas esperadas. Com um único
  "local certo" por pergunta, isso é o mesmo que hit rate@k.
- MRR (mean reciprocal rank): média de 1/posição do primeiro acerto (0 se não
  acertou). Diferencia "achou em 1º" (1,0) de "achou em 5º" (0,2), algo que o
  recall@5 sozinho não mostra.

Uso (de dentro de `backend/`, com o venv ativo):

    # 1) Avalia o índice atual (data/chroma), sem reindexar nada. Rápido.
    python3 evaluate.py

    # 2) Compara configurações: para cada combinação de chunk size e overlap,
    #    indexa os PDFs da pasta numa coleção TEMPORÁRIA e mede. O índice
    #    principal não é tocado.
    python3 evaluate.py --pdf-dir .. --chunk-sizes 100,200,300 --overlaps 0,40

    # Opções: --top-k 1,3,5,10   --eval outro_eval.json   --verbose
"""

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.embeddings import Embedder, get_embedder
from app.ingestion import PdfExtractionError
from app.rag.indexing import index_pdf
from app.rag.retrieval import retrieve
from app.store import EmbeddingModelMismatchError, SearchResult, VectorStore, get_vector_store

# ---------------------------------------------------------------------- #
# Formato do eval.json
# ---------------------------------------------------------------------- #


class ExpectedLocation(BaseModel):
    filename: str = Field(description="Nome do arquivo PDF, como aparece em GET /documents.")
    pages: list[int] = Field(min_length=1, description="Qualquer uma destas páginas conta como acerto.")


class EvalQuestion(BaseModel):
    id: str
    question: str
    expected: list[ExpectedLocation] = Field(min_length=1)
    answer_hint: str | None = None


class EvalFile(BaseModel):
    description: str | None = None
    questions: list[EvalQuestion] = Field(min_length=1)


# ---------------------------------------------------------------------- #
# Métricas
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuestionOutcome:
    question: EvalQuestion
    # Posição (1 = primeiro) do primeiro chunk correto, ou None se não apareceu.
    first_hit_rank: int | None
    results: list[SearchResult]


@dataclass(frozen=True)
class ConfigResult:
    label: str
    num_chunks: int
    # Chunks maiores que o limite do modelo (None = desconhecido, índice atual).
    truncated_chunks: int | None
    index_seconds: float | None
    outcomes: list[QuestionOutcome]

    def recall_at(self, k: int) -> float:
        hits = sum(1 for o in self.outcomes if o.first_hit_rank is not None and o.first_hit_rank <= k)
        return hits / len(self.outcomes)

    def mrr(self) -> float:
        return sum(1 / o.first_hit_rank for o in self.outcomes if o.first_hit_rank) / len(self.outcomes)


def is_hit(result: SearchResult, expected: list[ExpectedLocation]) -> bool:
    """O chunk vem do arquivo esperado e seu intervalo de páginas cobre uma página esperada?

    Usamos o INTERVALO [page_start, page_end] porque um chunk que atravessa a
    quebra de página 8→9 contém, de fato, texto da página 9.
    """
    chunk = result.chunk
    return any(
        chunk.filename == exp.filename
        and any(chunk.page_start <= page <= chunk.page_end for page in exp.pages)
        for exp in expected
    )


def evaluate_store(
    label: str,
    questions: list[EvalQuestion],
    max_k: int,
    embedder: Embedder,
    store: VectorStore,
    index_seconds: float | None = None,
    truncated_chunks: int | None = None,
) -> ConfigResult:
    outcomes: list[QuestionOutcome] = []
    for q in questions:
        # Uma única busca com o maior k serve para todos os k menores: o
        # top-3 é simplesmente o começo do top-10.
        results = retrieve(q.question, max_k, embedder=embedder, store=store)
        rank = next((i for i, r in enumerate(results, start=1) if is_hit(r, q.expected)), None)
        outcomes.append(QuestionOutcome(question=q, first_hit_rank=rank, results=results))
    return ConfigResult(
        label=label,
        num_chunks=store.count(),
        truncated_chunks=truncated_chunks,
        index_seconds=index_seconds,
        outcomes=outcomes,
    )


# ---------------------------------------------------------------------- #
# Execução
# ---------------------------------------------------------------------- #


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Mede recall@k e MRR da recuperação.")
    parser.add_argument("--eval", type=Path, default=Path(__file__).parent / "eval.json",
                        help="arquivo de perguntas (padrão: eval.json)")
    parser.add_argument("--top-k", type=int_list, default=[1, 3, 5, 10],
                        help="valores de k separados por vírgula (padrão: 1,3,5,10)")
    parser.add_argument("--pdf-dir", type=Path,
                        help="pasta com os PDFs; ativa a comparação de configurações")
    parser.add_argument("--chunk-sizes", type=int_list, default=[settings.chunk_size],
                        help=f"ex.: 100,200,300 (padrão: {settings.chunk_size})")
    parser.add_argument("--overlaps", type=int_list, default=[settings.chunk_overlap],
                        help=f"ex.: 0,40 (padrão: {settings.chunk_overlap})")
    parser.add_argument("--verbose", action="store_true", help="mostrar o resultado de cada pergunta")
    args = parser.parse_args()
    # O aviso de truncamento por PDF vira a coluna "truncados" da tabela.
    logging.getLogger("app.rag.indexing").setLevel(logging.ERROR)

    eval_file = load_eval(args.eval)
    ks = sorted(set(args.top_k))
    print(f"{len(eval_file.questions)} perguntas de {args.eval.name}; k = {', '.join(map(str, ks))}")
    print(f"Modelo de embeddings: {settings.embedding_model}\n")
    embedder = get_embedder()

    if args.pdf_dir is None:
        try:
            store = get_vector_store()
        except EmbeddingModelMismatchError as exc:
            sys.exit(f"Erro: {exc}")
        if store.count() == 0:
            sys.exit("O índice está vazio. Indexe PDFs ou use --pdf-dir.")
        results = [evaluate_store("índice atual", eval_file.questions, max(ks), embedder, store)]
    else:
        results = run_grid(args, eval_file, ks, embedder)

    print_table(results, ks)
    if args.verbose:
        print_details(results)
    else:
        print_misses(results, max(ks))


def run_grid(args: argparse.Namespace, eval_file: EvalFile, ks: list[int], embedder: Embedder) -> list[ConfigResult]:
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF em {args.pdf_dir}")
    expected_files = {e.filename for q in eval_file.questions for e in q.expected}
    missing = expected_files - {p.name for p in pdfs}
    if missing:
        print(f"AVISO: PDFs esperados no eval.json que não estão em {args.pdf_dir}: {sorted(missing)}\n")

    combos = [(size, ov) for size in args.chunk_sizes for ov in args.overlaps]
    results: list[ConfigResult] = []
    # Coleções temporárias, apagadas no fim: o índice principal fica intacto.
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        for i, (size, overlap) in enumerate(combos, start=1):
            label = f"size={size} overlap={overlap}"
            if overlap >= size:
                print(f"[{i}/{len(combos)}] {label}: ignorado (overlap >= chunk size)")
                continue
            print(f"[{i}/{len(combos)}] {label}: indexando {len(pdfs)} PDF(s)...", flush=True)
            store = VectorStore(Path(tmp), f"eval-{size}-{overlap}", embedder.model_name)
            started = time.perf_counter()
            truncated = 0
            for pdf in pdfs:
                try:
                    report = index_pdf(pdf, embedder=embedder, store=store, chunk_size=size, chunk_overlap=overlap)
                    truncated += report.truncated_chunks
                except PdfExtractionError as exc:
                    print(f"    ignorando {pdf.name}: {exc}")
            elapsed = time.perf_counter() - started
            results.append(
                evaluate_store(label, eval_file.questions, max(ks), embedder, store, elapsed, truncated)
            )
    print()
    return results


def load_eval(path: Path) -> EvalFile:
    try:
        return EvalFile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"Arquivo não encontrado: {path}")
    except ValidationError as exc:
        sys.exit(f"{path} inválido:\n{exc}")


def int_list(value: str) -> list[int]:
    try:
        return [int(v) for v in value.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"esperado inteiros separados por vírgula: '{value}'")


# ---------------------------------------------------------------------- #
# Saída
# ---------------------------------------------------------------------- #


def print_table(results: list[ConfigResult], ks: list[int]) -> None:
    headers = ["configuração", "chunks", "truncados", *[f"recall@{k}" for k in ks], "MRR", "indexação"]
    rows = [
        [
            r.label,
            str(r.num_chunks),
            str(r.truncated_chunks) if r.truncated_chunks is not None else "-",
            *[f"{r.recall_at(k):.2f}" for k in ks],
            f"{r.mrr():.3f}",
            f"{r.index_seconds:.1f}s" if r.index_seconds is not None else "-",
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for row in rows:
        print("  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths))))
    print()


def print_misses(results: list[ConfigResult], max_k: int) -> None:
    for r in results:
        misses = [o for o in r.outcomes if o.first_hit_rank is None]
        if misses:
            ids = ", ".join(o.question.id for o in misses)
            print(f"{r.label}: fora do top-{max_k} -> {ids}")
    print("(use --verbose para ver o que foi recuperado em cada pergunta)")


def print_details(results: list[ConfigResult]) -> None:
    for r in results:
        print(f"=== {r.label}")
        for o in r.outcomes:
            rank = f"acerto em #{o.first_hit_rank}" if o.first_hit_rank else "NÃO encontrado"
            expected = "; ".join(f"{e.filename} p. {e.pages}" for e in o.question.expected)
            print(f"  [{o.question.id}] {rank}   (esperado: {expected})")
            for i, res in enumerate(o.results[:5], start=1):
                c = res.chunk
                pages = f"p. {c.page_start}" if c.page_start == c.page_end else f"p. {c.page_start}–{c.page_end}"
                mark = "✓" if is_hit(res, o.question.expected) else " "
                print(f"      {mark} #{i} {res.score:.4f}  {c.filename} {pages}")
        print()


if __name__ == "__main__":
    main()
