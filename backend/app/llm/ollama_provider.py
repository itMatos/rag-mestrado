"""Provedor Ollama (LLM local), via API HTTP do Ollama.

Usamos o endpoint `/api/chat` diretamente com httpx, em vez da biblioteca
`ollama`, para que o protocolo fique visível: com `"stream": true` o Ollama
responde com uma linha JSON por pedaço de texto (NDJSON), por exemplo

    {"message": {"role": "assistant", "content": "Os"}, "done": false}
    {"message": {"role": "assistant", "content": " autores"}, "done": false}
    ...
    {"message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "stop", ...}
"""

import json
from collections.abc import Iterator

import httpx

from app.llm.base import LLMError, LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        max_tokens: int,
        num_ctx: int,
        temperature: float,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def stream(self, system: str, user: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise LLMError(self._explain_http_error(response))
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise LLMError(f"Erro do Ollama: {data['error']}")
                    if content := data.get("message", {}).get("content"):
                        yield content
                    if data.get("done"):
                        if data.get("done_reason") == "length":
                            yield "\n\n[Resposta interrompida: limite de LLM_MAX_TOKENS atingido.]"
                        break
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Não foi possível conectar ao Ollama em {self.base_url}. "
                "Ele está rodando? (abra o app Ollama ou rode `ollama serve`)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"O Ollama não respondeu em {self.timeout_seconds:.0f}s. Na primeira chamada "
                "o modelo é carregado na memória; tente de novo ou aumente LLM_TIMEOUT_SECONDS."
            ) from exc

    def _explain_http_error(self, response: httpx.Response) -> str:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        if response.status_code == 404:
            return f"Modelo '{self.model}' não está instalado no Ollama. Rode: ollama pull {self.model}"
        return f"Erro do Ollama ({response.status_code}): {detail}"
