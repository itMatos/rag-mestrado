"""Demonstração isolada de embeddings: similaridade entre frases.

Não usa o índice — só o modelo. Serve para criar intuição sobre:
- busca entre idiomas (pergunta em PT, texto em EN);
- o efeito dos prefixos "query:"/"passage:" do e5;
- a faixa típica de scores do modelo (o que é "alto" e o que é "baixo").

Uso:
    python3 -m scripts.similarity_demo
    python3 -m scripts.similarity_demo "sua pergunta" "frase 1" "frase 2" ...
"""

import sys

import numpy as np

from app.embeddings import get_embedder

DEFAULT_QUERY = "Quais são os riscos de usar código gerado por IA?"
DEFAULT_PASSAGES = [
    "AI-generated code can introduce long-term maintenance costs into software projects.",
    "Código gerado por inteligência artificial pode introduzir defeitos difíceis de manter.",
    "The study collected commits from open-source repositories on GitHub.",
    "A receita leva três ovos, farinha e açúcar.",
]


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    passages = sys.argv[2:] or DEFAULT_PASSAGES

    embedder = get_embedder()
    print(f"Modelo: {embedder.model_name} ({embedder.dimension} dimensões)")
    print(f'Prefixos: consulta="{embedder.prefixes.query}" passagem="{embedder.prefixes.passage}"\n')

    query_vec = embedder.embed_query(query)
    passage_vecs = embedder.embed_passages(passages)
    print(f"Norma do vetor da pergunta: {np.linalg.norm(query_vec):.4f}  (normalizado => 1.0)\n")

    # Como os vetores estão normalizados, o produto escalar JÁ É o cosseno.
    scores = passage_vecs @ query_vec
    print(f'Pergunta: "{query}"\n')
    for score, passage in sorted(zip(scores, passages), reverse=True):
        print(f"  {score:.4f}  {passage}")

    # Mesmo cálculo SEM os prefixos, para ver a diferença.
    if embedder.prefixes.query or embedder.prefixes.passage:
        raw = embedder.embed_without_prefix([query, *passages])
        print("\nSem prefixos (só para comparação; o pipeline sempre usa os prefixos):")
        for score, passage in sorted(zip(raw[1:] @ raw[0], passages), reverse=True):
            print(f"  {score:.4f}  {passage}")


if __name__ == "__main__":
    main()
