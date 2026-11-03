"""Interface comum a todos os provedores de LLM.

O resto do pipeline só conhece `LLMProvider`. Trocar Ollama por Anthropic (ou
adicionar outro provedor) não muda nada em `app/rag` nem na API: basta uma nova
subclasse e uma linha em `app/llm/__init__.py`.

A interface é propositalmente mínima — um prompt de sistema e uma mensagem do
usuário, sem histórico de conversa. Cada pergunta é respondida de forma
independente, com o contexto recuperado para ELA. Um chat com memória exigiria
decidir também o que recuperar quando a pergunta depende das anteriores
("e no outro artigo?"), o que é um problema à parte.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMError(Exception):
    """Falha ao chamar o LLM, com uma mensagem pensada para quem vai corrigir."""


class LLMProvider(ABC):
    #: Identificador curto ("ollama", "anthropic"), usado em logs e na API.
    name: str
    model: str

    @abstractmethod
    def stream(self, system: str, user: str) -> Iterator[str]:
        """Gera a resposta em pedaços de texto, à medida que o modelo produz.

        Streaming importa para a experiência: a primeira palavra aparece em
        ~1 s, enquanto a resposta completa pode levar 10–30 s.

        Deve lançar `LLMError` (e não exceções do SDK) em caso de falha.
        """

    def complete(self, system: str, user: str) -> str:
        """Resposta inteira de uma vez (útil em scripts e testes)."""
        return "".join(self.stream(system, user))
