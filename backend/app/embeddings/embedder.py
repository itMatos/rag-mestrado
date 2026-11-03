"""Geração de embeddings com sentence-transformers, rodando localmente.

Um embedding é um vetor (ex.: 768 números) que representa o significado de um
texto. Textos com significado parecido viram vetores que apontam para direções
parecidas, e a busca semântica é só "encontrar os vetores mais próximos do
vetor da pergunta".

Três regras que esta classe garante:

1. O MESMO modelo para indexar e para consultar. Cada modelo cria seu próprio
   "espaço" de vetores; comparar um vetor do modelo A com um do modelo B é como
   comparar coordenadas de dois mapas diferentes — os números existem, mas não
   significam nada. Por isso o nome do modelo é gravado na coleção do ChromaDB
   e conferido a cada abertura.

2. Os PREFIXOS que o modelo exige. A família e5 foi treinada com "query: " na
   frente das perguntas e "passage: " na frente dos documentos — é assim que o
   modelo sabe que uma pergunta curta deve ficar perto da passagem que a
   RESPONDE, e não de outra pergunta parecida. Sem os prefixos a qualidade da
   busca cai de forma perceptível. O bge-m3 não usa prefixos.

3. Vetores NORMALIZADOS (comprimento 1). Com isso, a similaridade de cosseno
   vira um simples produto escalar e fica sempre entre -1 e 1, comparável
   entre consultas. Sem normalizar, textos longos tendem a gerar vetores
   "maiores", e a busca passaria a favorecer o tamanho em vez do significado.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prefixes:
    query: str
    passage: str


# Prefixos exigidos por cada família de modelo (ver o "model card" no
# Hugging Face de cada um). A chave é comparada com o início do nome do modelo.
_KNOWN_PREFIXES: dict[str, Prefixes] = {
    "intfloat/multilingual-e5": Prefixes(query="query: ", passage="passage: "),
    "intfloat/e5": Prefixes(query="query: ", passage="passage: "),
    "BAAI/bge-m3": Prefixes(query="", passage=""),
}


def prefixes_for(model_name: str) -> Prefixes:
    for name_start, prefixes in _KNOWN_PREFIXES.items():
        if model_name.startswith(name_start):
            return prefixes
    logger.warning(
        "Modelo '%s' sem prefixos conhecidos; usando nenhum. Confira o model card.",
        model_name,
    )
    return Prefixes(query="", passage="")


class Embedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.prefixes = prefixes_for(model_name)
        # Na primeira execução o modelo é baixado do Hugging Face para
        # ~/.cache/huggingface; depois roda 100% offline.
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return int(self._model.get_embedding_dimension() or 0)

    @property
    def max_tokens(self) -> int:
        """Limite de entrada do modelo; o que passar disso é truncado."""
        return int(self._model.max_seq_length)

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embeddings dos chunks (lado "documento"). Shape: (len(texts), dimension)."""
        return self._encode([self.prefixes.passage + t for t in texts], show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        """Embedding da pergunta (lado "consulta"). Shape: (dimension,)."""
        return self._encode([self.prefixes.query + text], show_progress=False)[0]

    def embed_without_prefix(self, texts: list[str]) -> np.ndarray:
        """Embeddings SEM prefixo. Só para experimentos (ver similarity_demo);
        o pipeline nunca usa isto."""
        return self._encode(texts, show_progress=False)

    def count_tokens(self, text: str, passage: bool = True) -> int:
        """Quantos tokens o modelo vê para este texto (com prefixo e tokens especiais).

        Útil para conferir se o CHUNK_SIZE em palavras cabe no limite do modelo.
        """
        prefix = self.prefixes.passage if passage else self.prefixes.query
        return len(self._model.tokenizer(prefix + text, verbose=False)["input_ids"])

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )


@lru_cache
def get_embedder() -> Embedder:
    """Instância única: carregar o modelo leva segundos e ocupa ~1 GB de RAM,
    então carregamos uma vez por processo e reaproveitamos."""
    settings = get_settings()
    return Embedder(
        settings.embedding_model,
        device=settings.embedding_device or None,
        batch_size=settings.embedding_batch_size,
    )
