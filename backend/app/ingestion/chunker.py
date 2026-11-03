"""Chunking por tamanho fixo (em palavras) com overlap.

Por que dividir em chunks?
- O modelo de embeddings tem limite de entrada (ex.: 512 tokens) e trunca o resto.
- Um embedding é a "média" do significado do texto. O vetor de um artigo
  inteiro mistura dezenas de assuntos e não fica perto de nenhuma pergunta
  específica; o vetor de um trecho de ~300 palavras é focado o suficiente para
  casar com uma pergunta.
- O LLM recebe só os top-k chunks. Chunks menores = mais fontes diferentes
  cabendo no prompt; chunks maiores = mais contexto em volta de cada achado.
  Esse é o principal trade-off para você experimentar.

Estratégia: o documento vira uma única sequência de palavras, e cada palavra
lembra de qual página veio. Uma janela de `chunk_size` palavras desliza sobre
essa sequência, andando `chunk_size - overlap` palavras por vez. Como cada
palavra carrega sua página, cada chunk sabe exatamente em que página começa e
termina — inclusive quando atravessa a quebra de página.
"""

from app.ingestion.models import Chunk, ExtractedDocument


def window_bounds(num_words: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Calcula os intervalos [início, fim) de cada janela.

    Separado do resto para ficar fácil de testar e de entender isoladamente.

    Exemplo com 10 palavras, chunk_size=4, overlap=1 (passo = 3):
        [0, 4), [3, 7), [6, 10)
    A palavra 3 aparece no fim do 1º chunk e no início do 2º; a 6, no 2º e no 3º.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap deve estar entre 0 e chunk_size - 1")

    step = chunk_size - overlap
    bounds: list[tuple[int, int]] = []
    for start in range(0, num_words, step):
        end = min(start + chunk_size, num_words)
        bounds.append((start, end))
        # Paramos assim que uma janela alcança o fim. Sem isso, a próxima
        # janela seria composta só de palavras já contidas no overlap deste
        # chunk — um chunk 100% redundante.
        if end == num_words:
            break
    return bounds


def chunk_document(doc: ExtractedDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    """Divide um documento extraído em chunks com metadados de página."""
    words: list[str] = []
    word_pages: list[int] = []  # word_pages[i] = página da palavra words[i]
    for page in doc.pages:
        # `split()` sem argumento quebra em qualquer espaço em branco
        # (inclusive \n\n entre parágrafos). Perdemos a formatação, mas para o
        # embedding só o texto importa.
        page_words = page.text.split()
        words.extend(page_words)
        word_pages.extend([page.page_number] * len(page_words))

    chunks: list[Chunk] = []
    for position, (start, end) in enumerate(window_bounds(len(words), chunk_size, overlap)):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{position:04d}",
                doc_id=doc.doc_id,
                title=doc.title,
                filename=doc.filename,
                page_start=word_pages[start],
                page_end=word_pages[end - 1],
                position=position,
                text=" ".join(words[start:end]),
            )
        )
    return chunks
