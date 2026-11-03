"""Configuração central do backend.

Todos os parâmetros que você vai querer variar em experimentos (tamanho do chunk,
overlap, top-k, modelo de embeddings, provedor de LLM...) ficam aqui. Os valores
padrão podem ser sobrescritos por variáveis de ambiente ou pelo arquivo `backend/.env`
(veja `.env.example`). O nome da variável de ambiente é o nome do campo em
maiúsculas: `chunk_size` -> `CHUNK_SIZE`.

Novas configurações serão adicionadas a cada etapa do pipeline.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Diretório `backend/`, usado para resolver caminhos independentemente de onde
# o processo foi iniciado.
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Ingestão (extração + chunking)
    # ------------------------------------------------------------------ #

    # Tamanho do chunk em PALAVRAS. Usamos palavras (e não tokens) porque é uma
    # unidade que você consegue ver e contar, e não depende do tokenizador do
    # modelo. O limite real do modelo de embeddings é em tokens (512 no
    # multilingual-e5-base). Medido num artigo técnico em inglês: média de 1,7
    # tokens por palavra, chegando a 2,7 em trechos com tabelas e números. Com
    # 300 palavras, 11 de 38 chunks estouravam o limite; com 200, só 1. Chunks
    # maiores que o limite são TRUNCADOS na hora do embedding — o final do
    # chunk deixa de existir para a busca. O script de indexação avisa quando
    # isso acontece.
    chunk_size: int = Field(default=200, gt=0)

    # Quantas palavras do fim de um chunk são repetidas no início do próximo.
    # O overlap existe porque o corte por tamanho é "cego": uma frase ou ideia
    # pode ser partida ao meio na fronteira entre dois chunks. Repetindo um
    # trecho, a ideia fica inteira em pelo menos um deles. O custo é redundância
    # (mais chunks, mais armazenamento e resultados parecidos no top-k).
    chunk_overlap: int = Field(default=40, ge=0)

    # Se True, tentamos detectar o título "Referências"/"References" e descartar
    # tudo a partir dele. A lista de referências é cheia de títulos de outros
    # artigos e palavras-chave do tema, então "parece" muito relevante para a
    # busca vetorial, mas não contém resposta para nada — só poluiria o top-k.
    remove_references: bool = True

    # Onde ficam os dados locais (ChromaDB, uploads...). Ignorado pelo git.
    data_dir: Path = BACKEND_DIR / "data"

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    # Precisa ser MULTILÍNGUE: a pergunta vem em português e boa parte dos
    # artigos está em inglês. Um modelo multilíngue projeta "recuperação de
    # informação" e "information retrieval" em vetores próximos, então a busca
    # funciona entre idiomas. Opções testadas:
    #   intfloat/multilingual-e5-base  (~1,1 GB, 768 dim, máx. 512 tokens)
    #   intfloat/multilingual-e5-small (~470 MB, 384 dim, mais rápido)
    #   BAAI/bge-m3                    (~2,2 GB, 1024 dim, máx. 8192 tokens)
    # Trocar o modelo exige REINDEXAR tudo (veja app/store/vector_store.py).
    embedding_model: str = "intfloat/multilingual-e5-base"

    # "cpu", "mps" (GPU do Mac Apple Silicon), "cuda"... Vazio = automático.
    embedding_device: str | None = None

    # Quantos chunks são enviados ao modelo de uma vez na indexação. Maior =
    # mais rápido, até onde a memória aguentar.
    embedding_batch_size: int = Field(default=16, gt=0)

    # ------------------------------------------------------------------ #
    # Banco vetorial (ChromaDB) e recuperação
    # ------------------------------------------------------------------ #

    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"
    collection_name: str = "articles"

    # Quantos chunks a busca devolve (e, na etapa 3, quantos vão para o
    # prompt do LLM). Poucos = risco de a resposta ficar de fora; muitos =
    # mais ruído e prompt mais caro/lento.
    top_k: int = Field(default=5, gt=0, le=50)

    # ------------------------------------------------------------------ #
    # LLM (geração)
    # ------------------------------------------------------------------ #

    # Qual implementação de app/llm usar: "ollama" (local, gratuito) ou
    # "anthropic" (API, pago, respostas melhores).
    llm_provider: Literal["ollama", "anthropic"] = "ollama"

    # Limite de tokens da resposta. Folgado de propósito: se for atingido, a
    # resposta é cortada no meio. Nos modelos Claude atuais o raciocínio
    # interno ("thinking") também conta para este limite.
    llm_max_tokens: int = Field(default=16000, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Anthropic ---
    # Se vazio, o SDK procura a variável de ambiente ANTHROPIC_API_KEY.
    # (Valores do .env NÃO viram variáveis de ambiente, por isso o campo existe.)
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    # Quanto o modelo "pensa" antes de responder. Mais esforço = respostas
    # mais cuidadosas, porém mais lentas e caras.
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    # Modelos de ~7–8B rodam bem com 16 GB de RAM. Para português e para
    # seguir instruções de citação, qwen2.5:7b costuma ir bem; llama3.1:8b e
    # gemma2:9b são alternativas.
    ollama_model: str = "qwen2.5:7b"
    # Janela de contexto, em tokens. O padrão do Ollama é pequeno, e um prompt
    # maior que a janela é truncado SILENCIOSAMENTE pelo início — justamente
    # onde ficam as instruções. top-5 chunks + instruções ≈ 3 mil tokens.
    ollama_num_ctx: int = Field(default=8192, gt=0)
    # Temperatura baixa = respostas mais "conservadoras", que se apegam ao
    # contexto. Para RAG queremos fidelidade, não criatividade.
    ollama_temperature: float = Field(default=0.1, ge=0)

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    # Origens que o navegador pode usar para chamar a API. 4200 é a porta
    # padrão do `ng serve`. No .env, use JSON: CORS_ORIGINS=["http://localhost:4200"]
    cors_origins: list[str] = ["http://localhost:4200"]
    # Tamanho máximo de cada PDF enviado por upload.
    max_upload_mb: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        # Com overlap >= chunk_size a janela nunca andaria para frente.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) deve ser menor que "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Retorna a configuração (lida uma única vez e reaproveitada)."""
    return Settings()
