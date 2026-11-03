"""Montagem do prompt de geração: o "A" (augmented) do RAG.

O LLM não "sabe" nada dos seus artigos: ele só vê o que colocamos no prompt.
Por isso o prompt tem duas partes com papéis distintos:

- SISTEMA: as regras, iguais para toda pergunta. Responder só com base no
  contexto, citar as fontes num formato fixo, admitir quando não sabe.
- USUÁRIO: os trechos recuperados para ESTA pergunta + a pergunta.

Detalhes que fazem diferença:
- Cada trecho já vem com a citação pronta (ex.: [Título, p. 9]). Pedir ao
  modelo que "monte" a citação a partir de metadados soltos gera mais erros
  (página trocada, título abreviado) do que pedir que ele COPIE um rótulo.
- Os trechos ficam delimitados por tags XML. Assim o modelo distingue
  claramente o que é fonte, o que é instrução e o que é a pergunta — e um
  trecho que por acaso contenha uma frase imperativa não vira instrução.
- Os trechos vão em ordem de relevância (o mais similar primeiro). Modelos
  tendem a prestar mais atenção ao início e ao fim do contexto.
- Existe uma frase exata para "não encontrei". Isso torna a recusa
  verificável (ex.: na avaliação) e desencoraja o modelo a completar lacunas
  com conhecimento próprio, que é a principal fonte de alucinação em RAG.
"""

from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

from app.ingestion.models import Chunk
from app.store import SearchResult

NOT_FOUND_ANSWER = "Não encontrei essa informação nos documentos fornecidos."

SYSTEM_PROMPT = f"""Você é um assistente de pesquisa que ajuda um aluno de mestrado a consultar os artigos científicos da revisão bibliográfica dele.

Regras:
1. Responda SOMENTE com base nos trechos fornecidos dentro de <trechos>. Não use conhecimento próprio, mesmo que você saiba a resposta.
2. Cite a fonte de cada afirmação logo após ela, copiando exatamente o rótulo de citação do trecho usado, no formato [Título, p. X]. Se uma afirmação vier de mais de um trecho, cite todos: [Título A, p. 3] [Título B, p. 7].
3. Nunca invente citações, páginas ou títulos que não estejam nos rótulos fornecidos.
4. Se os trechos não contêm a resposta, responda exatamente: "{NOT_FOUND_ANSWER}" Se contêm só uma parte, responda essa parte e diga explicitamente o que não foi encontrado.
5. Responda sempre em português, mesmo quando os trechos estiverem em inglês. Termos técnicos podem ficar no original quando a tradução não for consagrada.
6. Seja direto e objetivo. Não comece com frases como "Com base nos trechos fornecidos"."""


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


def citation_label(chunk: Chunk) -> str:
    """Rótulo que o modelo deve copiar: [Título, p. X] ou [Título, p. X–Y]."""
    pages = (
        f"p. {chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"p. {chunk.page_start}–{chunk.page_end}"
    )
    return f"[{chunk.title}, {pages}]"


def build_prompt(question: str, results: list[SearchResult]) -> Prompt:
    passages = "\n\n".join(
        f"<trecho id={quoteattr(str(i))} citacao={quoteattr(citation_label(r.chunk))}>\n"
        f"{escape(r.chunk.text)}\n"
        f"</trecho>"
        for i, r in enumerate(results, start=1)
    )
    user = (
        f"<trechos>\n{passages}\n</trechos>\n\n"
        f"<pergunta>\n{escape(question)}\n</pergunta>"
    )
    return Prompt(system=SYSTEM_PROMPT, user=user)
