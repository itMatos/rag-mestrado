"""Extração de texto de PDFs com PyMuPDF, preservando o número da página.

Pipeline aplicado a cada PDF:

1. Abre o arquivo e calcula um `doc_id` a partir do conteúdo (hash).
2. Detecta o título (metadados do PDF -> maior fonte da 1ª página -> nome do arquivo).
3. Extrai o texto página a página, em blocos (≈ parágrafos).
4. Remove cabeçalhos/rodapés repetidos e números de página soltos.
5. Junta as linhas de cada bloco, desfazendo hifenização de fim de linha.
6. Corta a seção de referências, se encontrada.

Cada etapa de limpeza existe porque o texto vira embedding: lixo que se repete
em todas as páginas (nome do periódico, DOI no rodapé) "puxa" todos os chunks
para perto uns dos outros no espaço vetorial e piora a busca.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from app.ingestion.models import ExtractedDocument, PageText

# Títulos comuns da seção de referências, em português e inglês. Aceita
# numeração opcional ("7. Referências", "VI References") e dois-pontos no fim.
# A linha inteira precisa ser só o título — "as referências citadas mostram..."
# no meio de um parágrafo não deve disparar o corte.
_REFERENCES_HEADING = re.compile(
    r"""^\s*
        (?:(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+)?      # numeração opcional
        (?:
            refer[eê]ncias(?:\s+bibliogr[aá]ficas)?
          | references(?:\s+cited)?
          | bibliograph(?:y|ie)
          | bibliografia
          | literature\s+cited
          | works\s+cited
        )
        \s*:?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Só aceitamos um título de referências que esteja depois desta fração do
# texto. Evita cortar o artigo inteiro por causa de um sumário no início.
_REFERENCES_MIN_POSITION = 0.3

# Uma linha que aparece em pelo menos esta fração das páginas é considerada
# cabeçalho/rodapé (nome do periódico, título corrido, licença...).
_REPEATED_LINE_MIN_PAGES_FRACTION = 0.5
_REPEATED_LINE_MAX_LENGTH = 120

_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:p\.?\s*)?\d{1,4}\s*$", re.IGNORECASE)

# Títulos de metadados que são claramente lixo gerado por ferramentas.
_BAD_METADATA_TITLE = re.compile(
    r"(\.(docx?|pdf|tex|dvi)$|^microsoft word|^untitled|^title$|^document\d*$)",
    re.IGNORECASE,
)

# Ligaturas tipográficas aparecem em muitos PDFs como um único caractere ("ﬁ").
# Para o modelo de embeddings, "ﬁnal" e "final" seriam palavras diferentes.
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


class PdfExtractionError(Exception):
    """O PDF não pôde ser lido ou não tem texto extraível."""


def compute_doc_id(data: bytes) -> str:
    """Id estável derivado do CONTEÚDO do arquivo (não do nome).

    O mesmo PDF enviado duas vezes (mesmo que renomeado) gera o mesmo id, o
    que permite detectar duplicatas e reindexar de forma idempotente.
    16 caracteres hex (64 bits) são mais que suficientes para uma biblioteca
    pessoal de artigos.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def extract_pdf(path: Path, remove_references: bool = True) -> ExtractedDocument:
    """Lê um PDF do disco. Atalho para `extract_pdf_bytes`."""
    return extract_pdf_bytes(path.read_bytes(), path.name, remove_references)


def extract_pdf_bytes(
    data: bytes, filename: str, remove_references: bool = True
) -> ExtractedDocument:
    """Extrai o texto limpo, página a página, de um PDF em memória.

    Recebe bytes (e não um caminho) porque na API o arquivo chega por upload.
    """
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF lança tipos variados para arquivo inválido
        raise PdfExtractionError(f"Não foi possível abrir '{filename}' como PDF: {exc}") from exc

    with pdf:
        num_pages = pdf.page_count
        title = _detect_title(pdf, filename)
        raw_pages = [_extract_page_blocks(page) for page in pdf]

    raw_pages = _remove_repeated_lines(raw_pages)
    pages = [
        PageText(page_number=i + 1, text=_blocks_to_text(blocks))
        for i, blocks in enumerate(raw_pages)
    ]

    references_page: int | None = None
    if remove_references:
        pages, references_page = _cut_references(pages)

    # Páginas vazias (só figura, página em branco) não geram chunks.
    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise PdfExtractionError(
            f"'{filename}' não tem texto extraível. Se for um PDF escaneado "
            "(imagem), é preciso aplicar OCR antes — isso não é feito aqui."
        )

    return ExtractedDocument(
        doc_id=compute_doc_id(data),
        title=title,
        filename=filename,
        num_pages=num_pages,
        pages=pages,
        references_page=references_page,
    )


# ---------------------------------------------------------------------- #
# Título
# ---------------------------------------------------------------------- #


def _detect_title(pdf: pymupdf.Document, filename: str) -> str:
    """Heurística em cascata para descobrir o título do artigo.

    O título é usado na citação [Título, p. X], então vale o esforço. Ainda
    assim é uma heurística: confira com o script `inspect_chunks`.
    """
    meta_title = _normalize((pdf.metadata or {}).get("title") or "").strip()
    if 10 <= len(meta_title) <= 300 and not _BAD_METADATA_TITLE.search(meta_title):
        return meta_title

    if pdf.page_count > 0:
        font_title = _largest_font_text(pdf[0])
        if font_title:
            return font_title

    return Path(filename).stem


def _largest_font_text(page: pymupdf.Page) -> str | None:
    """Texto escrito no maior tamanho de fonte da página (normalmente o título).

    Percorre os tamanhos do maior para o menor e devolve o primeiro que forma
    um texto de tamanho plausível — assim pulamos letras capitulares gigantes
    ou um logotipo de uma palavra só.
    """
    spans_by_size: dict[float, list[str]] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    spans_by_size.setdefault(round(span["size"], 1), []).append(text)

    for size in sorted(spans_by_size, reverse=True):
        candidate = _normalize(" ".join(spans_by_size[size]))
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if 10 <= len(candidate) <= 300:
            return candidate
    return None


# ---------------------------------------------------------------------- #
# Texto das páginas
# ---------------------------------------------------------------------- #


def _extract_page_blocks(page: pymupdf.Page) -> list[list[str]]:
    """Devolve os blocos de texto da página, cada um como lista de linhas.

    Usamos o modo "blocks" porque cada bloco corresponde aproximadamente a um
    parágrafo. Mantemos a ordem do próprio PDF (sem `sort=True`): em artigos de
    duas colunas, ordenar por coordenada y misturaria as colunas linha a linha.
    """
    blocks: list[list[str]] = []
    # TEXT_MEDIABOX_CLIP ignora texto fora da área visível da página.
    for block in page.get_text("blocks", flags=pymupdf.TEXT_MEDIABOX_CLIP):
        block_type = block[6]
        if block_type != 0:  # 1 = imagem
            continue
        lines = [_normalize(line).strip() for line in block[4].splitlines()]
        lines = [line for line in lines if line]
        if lines:
            blocks.append(lines)
    return blocks


def _normalize(text: str) -> str:
    # NFC junta letra + acento combinante ("a" + "´" -> "á"), que alguns PDFs
    # produzem separados. Sem isso, "informação" poderia ter duas grafias em bytes.
    text = unicodedata.normalize("NFC", text)
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text.replace("­", "")  # hífen "suave" invisível


def _line_signature(line: str) -> str:
    # Troca dígitos por '#' para que "Revista X, p. 12" e "Revista X, p. 13"
    # sejam reconhecidos como o mesmo cabeçalho.
    return re.sub(r"\d+", "#", line.lower()).strip()


def _remove_repeated_lines(pages: list[list[list[str]]]) -> list[list[list[str]]]:
    """Remove cabeçalhos/rodapés repetidos e números de página isolados."""
    repeated: set[str] = set()
    if len(pages) >= 3:  # com poucas páginas não dá para distinguir repetição de acaso
        counts: Counter[str] = Counter()
        for blocks in pages:
            # `set` para contar cada assinatura no máximo uma vez por página.
            counts.update({_line_signature(line) for block in blocks for line in block})
        min_pages = max(3, int(len(pages) * _REPEATED_LINE_MIN_PAGES_FRACTION))
        repeated = {
            sig
            for sig, n in counts.items()
            if n >= min_pages and len(sig) <= _REPEATED_LINE_MAX_LENGTH
        }

    cleaned: list[list[list[str]]] = []
    for blocks in pages:
        new_blocks: list[list[str]] = []
        for block in blocks:
            kept = [
                line
                for line in block
                if _line_signature(line) not in repeated and not _PAGE_NUMBER_LINE.match(line)
            ]
            if kept:
                new_blocks.append(kept)
        cleaned.append(new_blocks)
    return cleaned


def _blocks_to_text(blocks: list[list[str]]) -> str:
    """Junta as linhas de cada bloco em um parágrafo; blocos separados por linha em branco."""
    paragraphs = [
        _join_lines(part) for lines in blocks for part in _split_on_references_heading(lines)
    ]
    return "\n\n".join(paragraphs)


def _split_on_references_heading(lines: list[str]) -> list[list[str]]:
    """Isola o título "Referências" em um parágrafo próprio.

    Em muitos PDFs o título vem no mesmo bloco da primeira referência
    ("References" + "[1] Smith, J. ..."). Se juntássemos as linhas, o título
    sumiria no meio do parágrafo e `_cut_references` não o encontraria.
    """
    parts: list[list[str]] = [[]]
    for line in lines:
        if _REFERENCES_HEADING.match(line):
            parts.extend([[line], []])
        else:
            parts[-1].append(line)
    return [part for part in parts if part]


def _join_lines(lines: list[str]) -> str:
    """Junta linhas quebradas pelo layout, desfazendo hifenização de fim de linha.

    "recupe-" + "ração" -> "recuperação". Só juntamos sem hífen quando a linha
    seguinte começa com letra minúscula e a anterior não termina em sigla ou
    número; caso contrário mantemos o hífen e colamos sem espaço ("COVID-" +
    "19" -> "COVID-19", "AI-" + "generated" -> "AI-generated"). Efeito
    colateral conhecido: palavras compostas em minúsculas quebradas exatamente
    no hífen ("bem-" + "estar") perdem o hífen. É um erro raro e barato
    comparado a deixar milhares de palavras partidas.
    """
    text = lines[0]
    for line in lines[1:]:
        before_hyphen = text[-2:-1]
        if text.endswith("-") and line[:1].islower() and before_hyphen.islower():
            text = text[:-1] + line
        elif text.endswith("-"):
            text = text + line
        else:
            text = f"{text} {line}"
    return text


# ---------------------------------------------------------------------- #
# Referências
# ---------------------------------------------------------------------- #


def _cut_references(pages: list[PageText]) -> tuple[list[PageText], int | None]:
    """Corta o documento a partir do ÚLTIMO título de referências encontrado.

    Procuramos o último (e não o primeiro) porque menções anteriores tendem a
    ser falsos positivos, como um sumário. Consequência aceita: apêndices que
    venham depois das referências também são descartados.
    """
    total_chars = sum(len(p.text) for p in pages) or 1
    chars_before = 0
    match: tuple[int, int] | None = None  # (índice da página, offset no texto)

    for page_idx, page in enumerate(pages):
        offset = 0
        for paragraph in page.text.split("\n\n"):
            position = (chars_before + offset) / total_chars
            if position >= _REFERENCES_MIN_POSITION and _REFERENCES_HEADING.match(paragraph):
                match = (page_idx, offset)
            offset += len(paragraph) + 2
        chars_before += len(page.text)

    if match is None:
        return pages, None

    page_idx, offset = match
    cut_page = pages[page_idx]
    kept = pages[:page_idx]
    remaining_text = cut_page.text[:offset].strip()
    if remaining_text:
        kept.append(PageText(page_number=cut_page.page_number, text=remaining_text))
    return kept, cut_page.page_number

