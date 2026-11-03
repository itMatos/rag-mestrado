"""Estruturas de dados da etapa de ingestão.

São dataclasses simples (e não modelos Pydantic) porque circulam só dentro do
pipeline; os schemas Pydantic da API ficam em `app/api/`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Texto limpo de uma página do PDF."""

    # Número FÍSICO da página no arquivo, começando em 1 — o mesmo que o seu
    # leitor de PDF mostra. Não é o número impresso no periódico (ex.: "p. 345"),
    # que raramente vem como metadado confiável. Assim a citação [Título, p. X]
    # sempre aponta para uma página que você consegue abrir no arquivo.
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Resultado da extração de um PDF, antes do chunking."""

    doc_id: str
    title: str
    filename: str
    # Total de páginas do arquivo (inclusive as descartadas, como referências).
    num_pages: int
    # Páginas com texto útil, já limpas. Pode ter menos itens que `num_pages`.
    pages: list[PageText]
    # Página onde a seção de referências foi detectada e cortada (None se não
    # foi encontrada ou se a remoção estava desligada).
    references_page: int | None


@dataclass(frozen=True)
class Chunk:
    """Um trecho do documento: a unidade que vira embedding e é recuperada."""

    # Determinístico: "<doc_id>:<posição>". Reindexar o mesmo PDF com os mesmos
    # parâmetros gera os mesmos ids, o que facilita comparar e evita duplicatas.
    chunk_id: str
    doc_id: str
    title: str
    filename: str
    # Um chunk pode atravessar a quebra de página; guardamos o intervalo para
    # poder citar "p. 3" ou "p. 3–4".
    page_start: int
    page_end: int
    # Ordem do chunk dentro do documento (0, 1, 2...).
    position: int
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def metadata(self) -> dict[str, str | int]:
        """Metadados planos (sem o texto), no formato aceito pelo ChromaDB."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "filename": self.filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "position": self.position,
        }
