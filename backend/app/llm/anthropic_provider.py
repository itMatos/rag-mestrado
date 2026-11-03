"""Provedor Anthropic (API da Claude), via SDK oficial `anthropic`."""

from collections.abc import Iterator

import anthropic

from app.llm.base import LLMError, LLMProvider

# Modelos que aceitam o "fallback" do lado do servidor: se o modelo principal
# recusar a pergunta por precaução de segurança (um falso positivo é possível
# em artigos de biologia ou segurança, por exemplo), a própria API refaz a
# requisição em outro modelo, dentro da mesma chamada.
_SERVER_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")
_SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        max_tokens: int,
        effort: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        # Sem api_key, o SDK procura ANTHROPIC_API_KEY no ambiente.
        self._client = (
            anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
            if api_key
            else anthropic.Anthropic(timeout=timeout_seconds)
        )

    def stream(self, system: str, user: str) -> Iterator[str]:
        extra: dict = {}
        if self.model in _SERVER_FALLBACK_MODELS:
            extra = {"betas": [_SERVER_FALLBACK_BETA], "fallbacks": "default"}

        try:
            with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                # O prompt de sistema traz as REGRAS (estáveis); a mensagem do
                # usuário traz o contexto recuperado + a pergunta (variáveis).
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                **extra,
            ) as stream:
                # `text_stream` entrega só os pedaços de texto da resposta;
                # o raciocínio interno do modelo não aparece aqui.
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("ANTHROPIC_API_KEY inválida.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Modelo '{self.model}' não encontrado. Confira ANTHROPIC_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Limite de requisições da Anthropic atingido; tente de novo em instantes.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Erro da API Anthropic ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Sem conexão com a API da Anthropic.") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Erro do SDK da Anthropic: {exc}") from exc
        except TypeError as exc:
            # O SDK só descobre a falta de credenciais ao montar a requisição,
            # e sinaliza isso com TypeError.
            if "authentication" not in str(exc):
                raise
            raise LLMError(
                "Credenciais da Anthropic não encontradas. Defina ANTHROPIC_API_KEY "
                "no backend/.env ou no ambiente."
            ) from exc

        # O motivo de parada diz se a resposta terminou normalmente.
        if final.stop_reason == "refusal":
            yield "\n\n[O modelo se recusou a responder esta pergunta.]"
        elif final.stop_reason == "max_tokens":
            yield "\n\n[Resposta interrompida: limite de LLM_MAX_TOKENS atingido.]"
