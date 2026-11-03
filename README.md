# rag-mestrado

Sistema de RAG (Retrieval-Augmented Generation) para conversar com os artigos
científicos de uma revisão bibliográfica (PDFs em português e inglês). Responde
em português, **só** com base nos artigos, citando `[Título, p. X]`, e mostra
os trechos recuperados com score.

O objetivo é **aprender RAG**: cada etapa do pipeline é implementada à mão, sem
LangChain ou LlamaIndex, e os comentários do código explicam o porquê de cada
decisão.

- `backend/`: Python + FastAPI (este README).
- `frontend/`: Angular (ainda não iniciado).
- [`docs/api.md`](docs/api.md): contrato completo da API, referência para o frontend.

## Como funciona

```
INDEXAÇÃO (uma vez por PDF)
PDF ──► extração (PyMuPDF) ──► limpeza ──► chunks de N palavras ──► embeddings ──► ChromaDB
        texto + nº da página   cabeçalhos,   com overlap; cada       "passage: ..."   vetores +
                               hifenização,  chunk sabe suas          (e5, normaliz.)  metadados
                               referências   páginas

CONSULTA (a cada pergunta)
pergunta ──► embedding ("query: ...") ──► top-k chunks (cosseno) ──► prompt ──► LLM ──► resposta
                                                │                    regras +    Ollama    em streaming
                                                │                    trechos     ou        com citações
                                                └──────── fontes + scores ───────────────► (SSE)
```

| Etapa | Código | O que observar |
|---|---|---|
| Extração e chunking | `app/ingestion/` | Páginas preservadas, referências cortadas, overlap |
| Embeddings | `app/embeddings/` | Prefixos `query:`/`passage:`, vetores normalizados, limite de tokens |
| Banco vetorial | `app/store/` | ChromaDB persistente, distância de cosseno, modelo gravado na coleção |
| Recuperação | `app/rag/retrieval.py` | Mesmo modelo na indexação e na consulta |
| Prompt | `app/rag/prompt.py` | Regras, rótulos de citação prontos, frase fixa de "não encontrei" |
| LLM | `app/llm/` | Interface única, provedores Ollama e Anthropic |
| API | `app/api/`, `app/main.py` | Rotas, schemas Pydantic, SSE |
| Avaliação | `evaluate.py`, `eval.json` | recall@k e MRR por configuração |

## Estrutura

```
rag-mestrado/
├── backend/
│   ├── app/
│   │   ├── main.py            # app FastAPI, CORS, carga inicial
│   │   ├── config.py          # todas as configurações (lidas do .env)
│   │   ├── ingestion/         # pdf_extractor.py, chunker.py, models.py
│   │   ├── embeddings/        # embedder.py (sentence-transformers)
│   │   ├── store/             # vector_store.py (ChromaDB)
│   │   ├── llm/               # base.py, ollama_provider.py, anthropic_provider.py
│   │   ├── rag/               # indexing.py, retrieval.py, prompt.py, generation.py
│   │   └── api/               # documents.py, search.py, chat.py, schemas.py
│   ├── scripts/               # testes de cada etapa pela linha de comando
│   ├── evaluate.py            # avaliação recall@k
│   ├── eval.json              # perguntas de avaliação
│   ├── requirements.txt
│   ├── .env.example
│   └── data/                  # (gerado, fora do git) índice ChromaDB
├── docs/api.md
└── README.md
```

## Instalação

Requisitos: **Python 3.11 ou 3.12** (PyTorch e ChromaDB costumam demorar a publicar
pacotes para versões mais novas), ~3 GB de disco para os modelos, e 16 GB de RAM
se for usar o Ollama.

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Na primeira execução, o modelo de embeddings (~1,1 GB) é baixado para
`~/.cache/huggingface/`. Depois disso, tudo roda offline (exceto a API da Anthropic).

### LLM: escolha um provedor

**Ollama (local, gratuito)**, que é o padrão:

```bash
brew install ollama            # ou o app de https://ollama.com/download
brew services start ollama
ollama pull qwen2.5:7b         # ~4,7 GB
```

**Anthropic (API, paga)**: crie uma chave em https://console.anthropic.com e, no
`backend/.env`, defina `LLM_PROVIDER=anthropic` e `ANTHROPIC_API_KEY=sk-ant-...`.

> **Use `python3`**, não `python`. Se o seu `~/.zshrc` tiver `alias python=/usr/bin/python3`,
> o alias tem prioridade sobre o venv e o Python do sistema seria usado.

## Execução

Todos os comandos rodam de dentro de `backend/`, com o venv ativo.

### Pela linha de comando (uma etapa por vez)

```bash
# 1. Ingestão: o que a extração e o chunking produzem (não salva nada)
python3 -m scripts.inspect_chunks "artigo.pdf"                  # --pages, --full, --chunk-size, --overlap

# 2. Indexação e busca
python3 -m scripts.similarity_demo                              # intuição sobre embeddings
python3 -m scripts.index_pdfs "artigo.pdf" ~/pasta/com/pdfs/    # indexa
python3 -m scripts.index_pdfs --list                            # lista; --delete <doc_id>
python3 -m scripts.search "sua pergunta" --top-k 5              # top-k com score

# 3. Geração
python3 -m scripts.ask "sua pergunta" --dry-run                 # só o prompt, sem LLM
python3 -m scripts.ask "sua pergunta"                           # resposta + fontes
python3 -m scripts.ask "sua pergunta" --provider anthropic --show-prompt
```

### API

```bash
uvicorn app.main:app --reload
```

- Swagger: http://localhost:8000/docs
- Contrato completo (requests, responses, eventos SSE, erros): [`docs/api.md`](docs/api.md)

```bash
curl -F "files=@artigo.pdf" localhost:8000/documents
curl localhost:8000/documents
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"question": "...", "top_k": 5}'
curl -N -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "..."}'
```

### Avaliação

```bash
python3 evaluate.py                                             # índice atual
python3 evaluate.py --verbose                                   # detalhes por pergunta
python3 evaluate.py --pdf-dir ~/pasta/com/pdfs --chunk-sizes 100,200,300 --overlaps 0,40
```

`eval.json` lista perguntas com o arquivo e as páginas onde está a resposta:

```json
{
  "id": "ferramentas-analise",
  "question": "Quais ferramentas de análise estática os autores usaram?",
  "expected": [{"filename": "artigo.pdf", "pages": [4]}],
  "answer_hint": "ESLint, Pylint e Semgrep"
}
```

Um acerto é um chunk do arquivo esperado cujo intervalo de páginas inclui uma
página esperada. O **recall@k** é a fração de perguntas com um acerto entre os k
primeiros. O **MRR** é a média de 1/posição do primeiro acerto. A comparação
`--pdf-dir` indexa em coleções temporárias e não altera o índice principal.

## Configuração (`backend/.env`)

Todos os valores são opcionais (os padrões estão em `app/config.py`).

| Variável | Padrão | Para quê |
|---|---|---|
| `CHUNK_SIZE` | `200` | Palavras por chunk |
| `CHUNK_OVERLAP` | `40` | Palavras repetidas entre chunks vizinhos (< `CHUNK_SIZE`) |
| `REMOVE_REFERENCES` | `true` | Cortar a seção de referências |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modelo de embeddings (trocar exige reindexar) |
| `EMBEDDING_DEVICE` | automático | `cpu`, `mps` (Mac) ou `cuda` |
| `EMBEDDING_BATCH_SIZE` | `16` | Chunks por lote na indexação |
| `COLLECTION_NAME` | `articles` | Coleção do ChromaDB (use outra para manter índices paralelos) |
| `TOP_K` | `5` | Chunks recuperados e enviados ao LLM |
| `LLM_PROVIDER` | `ollama` | `ollama` ou `anthropic` |
| `LLM_MAX_TOKENS` | `16000` | Limite da resposta |
| `LLM_TIMEOUT_SECONDS` | `120` | Tempo máximo de espera pelo LLM |
| `ANTHROPIC_API_KEY` | — | Chave da API (ou variável de ambiente) |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Ex.: `claude-sonnet-5-5`, `claude-haiku-5-5` |
| `ANTHROPIC_EFFORT` | `medium` | `low` … `max`: quanto o modelo raciocina |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ex.: `qwen2.5:3b`, `llama3.1:8b` |
| `OLLAMA_NUM_CTX` | `8192` | Janela de contexto (o prompt é truncado se passar disso) |
| `OLLAMA_TEMPERATURE` | `0.1` | Baixa = mais fiel ao contexto |
| `CORS_ORIGINS` | `["http://localhost:4200"]` | Origens do frontend (lista JSON) |
| `MAX_UPLOAD_MB` | `50` | Tamanho máximo por PDF no upload |

Variáveis de ambiente têm prioridade sobre o `.env`, o que é útil para testes
pontuais: `TOP_K=10 python3 -m scripts.ask "..."`.

## Experimentos sugeridos

Para cada experimento, adicione ao `eval.json` pelo menos 20 perguntas
sobre vários artigos. **Com 3 perguntas, uma única pergunta muda o recall em
0,33**, e qualquer "conclusão" é ruído.

### 1. Tamanho do chunk

```bash
python3 evaluate.py --pdf-dir ~/pdfs --chunk-sizes 100,150,200,300 --overlaps 40 --verbose
```

- **Observe:** recall@1 vs. recall@5, e a coluna `truncados`.
- **O que esperar:** chunks pequenos são "focados" (bom para fatos pontuais, como
  números e nomes de ferramentas), mas perdem o contexto em volta. Chunks
  grandes misturam assuntos e diluem o embedding. Acima de ~250 palavras, o
  e5-base começa a **truncar** chunks: um recall alto com muitos truncados
  engana, porque o final desses chunks é invisível para a busca.
- **Também:** com chunks menores, o `/chat` envia menos texto ao LLM para o mesmo
  `top_k`. Isso deixa a resposta mais rápida no Ollama, mas com menos contexto.

### 2. Overlap

```bash
python3 evaluate.py --pdf-dir ~/pdfs --chunk-sizes 200 --overlaps 0,20,40,80
```

- **Observe:** recall e número de chunks.
- **O que esperar:** sem overlap, uma frase cortada na fronteira entre chunks
  pode não ficar inteira em nenhum deles. Overlap demais gera chunks
  quase repetidos, que ocupam vagas do top-k com o mesmo conteúdo (rode
  `scripts.search` e veja resultados vizinhos, como `:0040` e `:0041`).

### 3. top-k

```bash
python3 evaluate.py --top-k 1,2,3,5,10,20
python3 -m scripts.ask "sua pergunta" --top-k 2
python3 -m scripts.ask "sua pergunta" --top-k 10
```

- **Observe:** o recall@k sobe com k (sempre), mas a resposta não melhora na
  mesma proporção.
- **O que esperar:** com k baixo, a resposta pode ficar de fora. Com k alto, entra
  mais ruído e o prompt cresce. No Ollama, cada trecho a mais soma ~400 tokens
  e alguns segundos até a primeira palavra. Escolha o menor k com recall
  aceitável.

### 4. Modelo de embeddings

```bash
EMBEDDING_MODEL=intfloat/multilingual-e5-small python3 evaluate.py --pdf-dir ~/pdfs
EMBEDDING_MODEL=BAAI/bge-m3 python3 evaluate.py --pdf-dir ~/pdfs
```

- **Observe:** recall, tempo de indexação e a faixa de scores
  (`scripts.similarity_demo`).
- **O que esperar:** o `e5-small` é mais rápido e um pouco pior. O `bge-m3` é
  maior (~2,2 GB), aceita até 8192 tokens (chunks bem maiores sem truncar) e não
  usa prefixos. Para usar outro modelo no índice principal, defina também outro
  `COLLECTION_NAME` (ou apague `data/chroma/`) e reindexe.

### 5. Prefixos do e5

Rode `python3 -m scripts.similarity_demo` e compare as duas listas, com e sem
prefixo. Depois, edite `_KNOWN_PREFIXES` em `app/embeddings/embedder.py` para
`Prefixes("", "")`, reindexe numa coleção separada (`COLLECTION_NAME=sem-prefixo`)
e compare o `evaluate.py`. O resultado mostra o efeito de usar o modelo fora do
formato em que ele foi treinado.

### 6. Busca entre idiomas

Faça a mesma pergunta em português e em inglês (`scripts.search`) sobre um
artigo em inglês. Os scores e as posições mudam pouco? Isso mostra o
modelo multilíngue funcionando. Repita com um modelo só em inglês para ver a
diferença.

### 7. Fidelidade do LLM (alucinação)

```bash
python3 -m scripts.ask "Qual é a capital da França?"
python3 -m scripts.ask "pergunta cuja resposta NÃO está nos artigos, mas é do mesmo tema"
```

- **Observe:** o modelo responde exatamente "Não encontrei essa informação nos
  documentos fornecidos." ou completa com conhecimento próprio?
- **Compare:** `qwen2.5:3b` vs. `qwen2.5:7b` vs. `--provider anthropic`. Compare também
  `OLLAMA_TEMPERATURE=0.1` vs. `0.8`.
- O segundo caso (mesmo tema, resposta ausente) é o mais difícil, porque os trechos
  recuperados "parecem" relevantes.

### 8. Citações

Use `--show-prompt` e confira se cada `[Título, p. X]` da resposta existe nos
rótulos `citacao=` dos trechos e se a página realmente contém a afirmação.
Modelos menores às vezes trocam a página ou citam o trecho errado.

### 9. Remoção de referências

```bash
python3 -m scripts.inspect_chunks "artigo.pdf" --keep-references --limit 0
```

Indexe um artigo com `REMOVE_REFERENCES=false` numa coleção separada e busque
por um tema citado só nas referências. Os chunks das referências aparecem no
topo, cheios de títulos de outros artigos, mas sem resposta alguma.

## Solução de problemas

| Sintoma | Causa / solução |
|---|---|
| `ModuleNotFoundError: No module named 'pydantic'` com o venv ativo | `alias python` no `~/.zshrc`. Use `python3`. |
| `unrecognized arguments` ao passar um PDF | Nome com espaços: coloque o caminho entre aspas. |
| `A coleção 'articles' foi indexada com '...'` | `EMBEDDING_MODEL` mudou. Use outro `COLLECTION_NAME` ou apague `backend/data/chroma/` e reindexe. |
| `AVISO: ... chunks passam de 512 tokens` | Reduza `CHUNK_SIZE` (tabelas e números usam mais tokens por palavra). |
| `Não foi possível conectar ao Ollama` | `brew services start ollama`. |
| `Modelo '...' não está instalado no Ollama` | `ollama pull <modelo>`. |
| Primeira resposta muito lenta | O Ollama carrega o modelo na memória (~10–30 s). As seguintes são mais rápidas. |
| `Redirects are currently not supported in MacOs`, `unauthenticated requests to the HF Hub` | Avisos inofensivos do PyTorch e do Hugging Face. |
| `Referências: título da seção NÃO encontrado` | O artigo usa outro título (ex.: "Notes") ou não tem uma linha só com o título. Veja `_REFERENCES_HEADING` em `pdf_extractor.py`. |

## Limitações conhecidas

- **Sem OCR:** PDFs escaneados (imagem) são rejeitados.
- **Título** detectado por heurística: confira com `scripts.inspect_chunks`.
- **Hifenização:** compostos quebrados no fim da linha perdem o hífen
  (`Claude-` + `authored` → `Claudeauthored`, `bem-` + `estar` → `bemestar`).
- **Listagens de código** com sublinhado ondulado geram sequências de `::::` no texto.
- **Apêndices** depois das referências também são cortados.
- **Páginas físicas**, não a numeração impressa do periódico.
- **Sem histórico de conversa:** cada pergunta é respondida isoladamente.
- **Score sem limiar:** a busca sempre devolve os k mais próximos. A recusa de
  perguntas fora do tema depende do LLM obedecer ao prompt.
