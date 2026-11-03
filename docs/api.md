# API do backend — contrato

Referência para quem for construir o frontend. Descreve o código como ele está
(`backend/app/api/`). O Swagger em `http://localhost:8000/docs` é gerado a partir
dos mesmos schemas Pydantic e serve para testar as rotas interativamente.

- **Base URL (desenvolvimento):** `http://localhost:8000`
- **Formato:** JSON (UTF-8), exceto o upload (`multipart/form-data`) e o chat (`text/event-stream`).
- **Autenticação:** nenhuma.
- **CORS:** liberado para as origens em `CORS_ORIGINS` (padrão: `["http://localhost:4200"]`),
  com todos os métodos e headers. Outras origens são bloqueadas pelo navegador.

## Sumário

| Método | Rota | Descrição |
|---|---|---|
| `POST` | [`/documents`](#post-documents) | Envia e indexa um ou mais PDFs |
| `GET` | [`/documents`](#get-documents) | Lista os documentos indexados |
| `DELETE` | [`/documents/{doc_id}`](#delete-documentsdoc_id) | Remove um documento e seus chunks |
| `GET` | [`/documents/{doc_id}/chunks`](#get-documentsdoc_idchunks) | Chunks de um documento (depuração) |
| `POST` | [`/search`](#post-search) | Só a recuperação: top-k chunks com score |
| `POST` | [`/chat`](#post-chat) | Resposta do LLM em streaming (SSE) + fontes |
| `GET` | [`/health`](#get-health) | Estado e configuração do backend |

---

## Tipos comuns

### `DocumentOut`

| Campo | Tipo | Descrição |
|---|---|---|
| `doc_id` | string | Hash (16 caracteres hex) do **conteúdo** do PDF. O mesmo arquivo gera sempre o mesmo id, mesmo renomeado. |
| `title` | string | Título detectado (metadados do PDF → maior fonte da 1ª página → nome do arquivo). |
| `filename` | string | Nome do arquivo enviado. |
| `num_pages` | integer | Total de páginas do arquivo (incluindo as descartadas, como referências). |
| `num_chunks` | integer | Quantos chunks o documento gerou. |

### `SourceOut` (um trecho recuperado)

| Campo | Tipo | Descrição |
|---|---|---|
| `chunk_id` | string | `"<doc_id>:<posição com 4 dígitos>"`, ex.: `"4b04440831c01219:0041"`. |
| `doc_id` | string | Documento de origem. |
| `title` | string | Título do documento. |
| `filename` | string | Arquivo de origem. |
| `page_start` | integer | Página (física, começando em 1) onde o trecho começa. |
| `page_end` | integer | Página onde termina. Igual a `page_start` quando não atravessa a quebra de página. |
| `position` | integer | Ordem do chunk dentro do documento (0, 1, 2...). |
| `citation` | string | Rótulo usado nas respostas: `"[Título, p. 9]"` ou `"[Título, p. 3–4]"` (traço meia-risca `–`). |
| `score` | number | Similaridade de cosseno, arredondada em 4 casas. Maior = mais parecido. Com o modelo e5 os valores ficam numa faixa estreita (≈ 0,70–0,85): compare scores entre si, não com um limiar fixo. |
| `text` | string | Texto do chunk (palavras separadas por espaço simples, sem quebras de linha). |

> **Páginas:** são as páginas físicas do arquivo, como o leitor de PDF mostra
> (a 1ª página do arquivo é 1), e não a numeração impressa do periódico.

---

## `POST /documents`

Envia e indexa um ou mais PDFs. A indexação acontece **dentro da requisição**
(extração → chunking → embeddings → ChromaDB) e leva alguns segundos por PDF.
O frontend deve mostrar um indicador de progresso e não reenviar enquanto isso.

**Request:** `multipart/form-data` com um ou mais campos chamados **`files`**.

```bash
curl -F "files=@artigo1.pdf" -F "files=@artigo2.pdf" http://localhost:8000/documents
```

```ts
const form = new FormData();
for (const file of selectedFiles) form.append('files', file);
await fetch('http://localhost:8000/documents', { method: 'POST', body: form });
// Não defina Content-Type manualmente: o navegador inclui o boundary.
```

**Response 200** (`UploadResponse`): um resultado por arquivo, na ordem do envio.
**Falhas de arquivos individuais NÃO mudam o status HTTP**: a resposta é 200 e o
erro vem no item correspondente.

```json
{
  "results": [
    {
      "filename": "Debt Behind the AI Boom A.pdf",
      "status": "replaced",
      "document": {
        "doc_id": "4b04440831c01219",
        "title": "Debt Behind the AI Boom: A Large-Scale Empirical Study of AI-Generated Code in the Wild",
        "filename": "Debt Behind the AI Boom A.pdf",
        "num_pages": 13,
        "num_chunks": 57
      },
      "references_page": 11,
      "truncated_chunks": 1,
      "error": null
    },
    {
      "filename": "notas.txt",
      "status": "error",
      "document": null,
      "references_page": null,
      "truncated_chunks": 0,
      "error": "O arquivo não é um PDF."
    }
  ]
}
```

| Campo de `UploadResult` | Tipo | Descrição |
|---|---|---|
| `filename` | string | Nome enviado (ou `"sem_nome.pdf"` se ausente). |
| `status` | `"indexed"` \| `"replaced"` \| `"error"` | `indexed`: novo. `replaced`: já existia um documento com o mesmo `doc_id` (mesmo conteúdo) e ele foi reindexado; os chunks antigos foram apagados. `error`: não indexado. |
| `document` | `DocumentOut` \| null | `null` quando `status = "error"`. |
| `references_page` | integer \| null | Página onde a seção de referências foi detectada e cortada; `null` se não encontrada. |
| `truncated_chunks` | integer | Chunks maiores que o limite do modelo de embeddings (o final deles é ignorado na busca). |
| `error` | string \| null | Mensagem legível; presente só quando `status = "error"`. |

Mensagens de `error` possíveis:

| Situação | `error` |
|---|---|
| Arquivo maior que `MAX_UPLOAD_MB` (padrão 50) | `"Arquivo maior que o limite de 50 MB."` |
| Conteúdo não começa com `%PDF-` (a extensão é ignorada) | `"O arquivo não é um PDF."` |
| PDF corrompido | `"Não foi possível abrir '<nome>' como PDF: ..."` |
| PDF sem texto (escaneado) | `"'<nome>' não tem texto extraível. Se for um PDF escaneado (imagem), é preciso aplicar OCR antes — isso não é feito aqui."` |

**Erros HTTP:** `422` se nenhum campo `files` for enviado (ver [Erros](#erros)).

---

## `GET /documents`

Lista os documentos indexados, ordenados por título (sem diferenciar maiúsculas).

**Response 200** (`DocumentListResponse`):

```json
{
  "documents": [
    {
      "doc_id": "4b04440831c01219",
      "title": "Debt Behind the AI Boom: A Large-Scale Empirical Study of AI-Generated Code in the Wild",
      "filename": "Debt Behind the AI Boom A.pdf",
      "num_pages": 13,
      "num_chunks": 57
    }
  ],
  "total_chunks": 57
}
```

Com o índice vazio: `{"documents": [], "total_chunks": 0}`.

---

## `DELETE /documents/{doc_id}`

Remove o documento e todos os seus chunks. Irreversível (para voltar, reenvie o PDF).

**Response 200** (`DeleteResponse`):

```json
{ "doc_id": "4b04440831c01219", "deleted_chunks": 57 }
```

**Response 404:** `{"detail": "Documento '<doc_id>' não encontrado."}`

---

## `GET /documents/{doc_id}/chunks`

Todos os chunks de um documento, em ordem (`position` crescente). Pensado para
depurar o chunking.

**Response 200** (`DocumentChunksResponse`):

```json
{
  "document": { "doc_id": "4b04440831c01219", "title": "...", "filename": "...", "num_pages": 13, "num_chunks": 57 },
  "chunks": [
    {
      "chunk_id": "4b04440831c01219:0000",
      "position": 0,
      "page_start": 1,
      "page_end": 1,
      "word_count": 200,
      "token_count": 323,
      "truncated": false,
      "text": "Debt Behind the AI Boom: ..."
    }
  ]
}
```

| Campo de `ChunkOut` | Tipo | Descrição |
|---|---|---|
| `chunk_id`, `position`, `page_start`, `page_end`, `text` | | Como em `SourceOut`. |
| `word_count` | integer | Palavras do chunk (unidade de `CHUNK_SIZE`). |
| `token_count` | integer | Tokens que o modelo de embeddings vê (inclui o prefixo `passage: `). |
| `truncated` | boolean | `true` se `token_count` passa do limite do modelo (512 no e5-base). |

**Response 404:** `{"detail": "Documento '<doc_id>' não encontrado."}`

---

## `POST /search`

Só a recuperação, sem LLM. Útil para testar a busca e para mostrar "o que o sistema viu".

**Request** (`SearchRequest`, `application/json`):

| Campo | Tipo | Obrigatório | Regras |
|---|---|---|---|
| `question` | string | sim | Espaços nas pontas são removidos; depois disso, 1 a 2000 caracteres. |
| `top_k` | integer \| null | não | 1 a 50. Ausente/`null` = `TOP_K` do `.env` (padrão 5). |
| `doc_ids` | string[] \| null | não | Restringe a busca a estes documentos. Ausente, `null` ou `[]` = todos. Ids inexistentes simplesmente não retornam nada. |

```json
{ "question": "Qual porcentagem dos problemas introduzidos por IA ainda sobrevive?", "top_k": 3 }
```

**Response 200** (`SearchResponse`):

```json
{
  "question": "Qual porcentagem dos problemas introduzidos por IA ainda sobrevive?",
  "top_k": 3,
  "results": [
    {
      "chunk_id": "4b04440831c01219:0041",
      "doc_id": "4b04440831c01219",
      "title": "Debt Behind the AI Boom: A Large-Scale Empirical Study of AI-Generated Code in the Wild",
      "filename": "Debt Behind the AI Boom A.pdf",
      "page_start": 9,
      "page_end": 9,
      "position": 41,
      "citation": "[Debt Behind the AI Boom: A Large-Scale Empirical Study of AI-Generated Code in the Wild, p. 9]",
      "score": 0.8102,
      "text": "the amount of AI-introduced debt in real-world repositories ..."
    }
  ]
}
```

- `top_k` na resposta é o valor efetivamente usado (o do request ou o padrão).
- `results` vem ordenado do maior para o menor `score` e pode ter **menos** que
  `top_k` itens (índice pequeno, filtro por `doc_ids`). Com o índice vazio, `[]`.
- A busca **sempre** devolve os mais próximos, mesmo para perguntas sem relação
  com os artigos: não existe "nenhum resultado relevante" nesta rota.

**Erros HTTP:** `422` para `question` vazia/longa, `top_k` fora de 1–50 ou JSON inválido.

---

## `POST /chat`

Pipeline RAG completo: recupera os top-k chunks, monta o prompt e transmite a
resposta do LLM em **Server-Sent Events**. Cada pergunta é independente (não há
histórico de conversa).

**Request** (`ChatRequest`): mesmos campos e regras de [`POST /search`](#post-search).

```json
{ "question": "Qual porcentagem dos problemas introduzidos por IA ainda sobrevive?", "top_k": 5 }
```

**Response 200:** `Content-Type: text/event-stream; charset=utf-8`, com
`Cache-Control: no-cache` e `X-Accel-Buffering: no`.

### Formato dos eventos

Cada evento é exatamente:

```
event: <nome>\n
data: <JSON em uma linha>\n
\n
```

O `data` é **sempre um objeto JSON em uma única linha**. Quebras de linha da
resposta chegam como `\n` dentro da string JSON. Não há `id:` nem `retry:`.

### Sequência

```
token    0..N vezes    pedaços da resposta, na ordem
error    0 ou 1 vez    falha do LLM no meio da resposta
sources  exatamente 1  trechos usados como contexto
done     exatamente 1  fim do stream (sempre o último)
```

`sources` e `done` são enviados **sempre**, inclusive depois de um `error`.
Depois de `done` o servidor fecha a conexão.

| Evento | `data` | Descrição |
|---|---|---|
| `token` | `{"text": string}` | Próximo pedaço da resposta (de um caractere a algumas palavras). Concatene todos na ordem. |
| `error` | `{"message": string}` | Erro legível. Os `token` anteriores (se houver) formam uma resposta parcial. |
| `sources` | `{"sources": SourceOut[]}` | Os trechos enviados ao LLM, na ordem de relevância (mesmo formato de `/search`). Pode ser `[]`. |
| `done` | `{"provider": string, "model": string, "elapsed_ms": integer, "error": boolean}` | `provider`: `"ollama"` ou `"anthropic"`. `elapsed_ms`: tempo de geração (a partir do início do stream). `error`: `true` se houve um evento `error`. |

### Exemplo (encurtado)

```
event: token
data: {"text":"Cerca de "}

event: token
data: {"text":"22,7% dos problemas ainda sobrevivem [Debt Behind the AI Boom: A Large-Scale Empirical Study of AI-Generated Code in the Wild, p. 9]."}

event: sources
data: {"sources":[{"chunk_id":"4b04440831c01219:0041","doc_id":"4b04440831c01219","title":"Debt Behind the AI Boom: ...","filename":"Debt Behind the AI Boom A.pdf","page_start":9,"page_end":9,"position":41,"citation":"[Debt Behind the AI Boom: ..., p. 9]","score":0.8102,"text":"..."}]}

event: done
data: {"provider":"ollama","model":"qwen2.5:7b","elapsed_ms":10332,"error":false}
```

### Conteúdo da resposta (texto dos `token`)

- Em português, com citações no formato **`[Título, p. X]`** ou
  **`[Título, p. X–Y]`**, copiadas do campo `citation` das fontes. Para
  transformar citações em links, procure no texto as strings `citation` de
  `sources` (o modelo pode, raramente, errar o rótulo).
- Quando os trechos não contêm a resposta, o texto é exatamente
  **`Não encontrei essa informação nos documentos fornecidos.`**
- Se nenhum trecho for recuperado (índice vazio, ou `doc_ids` sem
  correspondência), o LLM **não** é chamado: chega um único `token` com essa
  frase, `sources` vazio e `done` com `elapsed_ms` ≈ 0.
- Se a resposta for cortada pelo limite de tokens, o último `token` é
  `"\n\n[Resposta interrompida: limite de LLM_MAX_TOKENS atingido.]"`. Com a
  Anthropic, uma recusa do modelo termina com
  `"\n\n[O modelo se recusou a responder esta pergunta.]"`. Ambos chegam como
  `token`, não como `error`.

### Mensagens de `error` possíveis

| Causa | `message` (início) |
|---|---|
| Ollama não está rodando | `Não foi possível conectar ao Ollama em http://localhost:11434...` |
| Modelo do Ollama não baixado | `Modelo 'qwen2.5:7b' não está instalado no Ollama. Rode: ollama pull qwen2.5:7b` |
| Ollama demorou mais que `LLM_TIMEOUT_SECONDS` | `O Ollama não respondeu em 120s...` |
| Sem `ANTHROPIC_API_KEY` | `Credenciais da Anthropic não encontradas...` |
| Chave inválida | `ANTHROPIC_API_KEY inválida.` |
| Limite de requisições | `Limite de requisições da Anthropic atingido...` |
| Outro erro da API Anthropic | `Erro da API Anthropic (<status>): ...` |
| Erro inesperado no servidor | `Erro interno ao gerar a resposta.` |

### Erros HTTP (antes do stream começar)

A validação e a busca acontecem **antes** de o stream abrir. Por isso, esses
erros chegam como respostas HTTP normais, em JSON, e não como eventos:
`422` (mesmas regras de `/search`) e `500` (falha inesperada na busca).

### Consumindo no navegador (Angular)

`EventSource` só faz `GET`, e esta rota é `POST`. Use `fetch` e leia o corpo
como stream. Os pedaços de rede **não** coincidem com os eventos: um pedaço pode
trazer meio evento ou vários. Por isso, acumule o texto num buffer e separe os
eventos por `\n\n`.

```ts
export interface Source {
  chunk_id: string; doc_id: string; title: string; filename: string;
  page_start: number; page_end: number; position: number;
  citation: string; score: number; text: string;
}

export interface ChatHandlers {
  onToken(text: string): void;
  onError(message: string): void;
  onSources(sources: Source[]): void;
  onDone(info: { provider: string; model: string; elapsed_ms: number; error: boolean }): void;
}

export async function chat(question: string, handlers: ChatHandlers, topK?: number, signal?: AbortSignal) {
  const res = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK ?? null }),
    signal, // AbortController.abort() cancela a geração no meio
  });
  if (!res.ok) {
    // 422/500: corpo JSON com "detail" (ver seção Erros)
    throw new Error(JSON.stringify(await res.json()));
  }

  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] ?? '{}');
      switch (event) {
        case 'token':   handlers.onToken(data.text); break;
        case 'error':   handlers.onError(data.message); break;
        case 'sources': handlers.onSources(data.sources); break;
        case 'done':    handlers.onDone(data); break;
      }
    }
  }
}
```

**Latência esperada:** com Ollama (`qwen2.5:7b`) num Mac M1, o primeiro `token`
leva cerca de 10 a 30 s (o prompt de ~2.300 tokens é processado antes de a
geração começar). Mostre um estado "pensando..." até o primeiro `token`.

---

## `GET /health`

**Response 200** (`HealthResponse`):

```json
{
  "status": "ok",
  "embedding_model": "intfloat/multilingual-e5-base",
  "llm_provider": "ollama",
  "llm_model": "qwen2.5:7b",
  "documents": 1,
  "chunks": 57
}
```

`status` é sempre `"ok"`: se o backend não consegue carregar o modelo ou o
índice, ele nem chega a iniciar. Este endpoint **não** verifica se o Ollama ou a
API da Anthropic estão acessíveis. Isso só aparece no primeiro `/chat`.

---

## Erros

| Status | Quando | Corpo |
|---|---|---|
| `404` | `doc_id` inexistente em `DELETE /documents/{id}` ou `GET /documents/{id}/chunks` | `{"detail": "Documento '<id>' não encontrado."}` |
| `405` | Método errado (ex.: `GET /chat`) | `{"detail": "Method Not Allowed"}` |
| `422` | Request inválida (campo faltando, fora dos limites, JSON malformado) | Formato padrão do FastAPI, abaixo |
| `500` | Erro inesperado | `Internal Server Error` (texto puro) |

Formato do `422`: `detail` é uma **lista**, com um item por problema. `loc`
indica onde está o problema:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "top_k"],
      "msg": "Input should be less than or equal to 50",
      "input": 51,
      "ctx": { "le": 50 }
    }
  ]
}
```

Exemplos de `type`: `missing` (campo obrigatório ausente, ex.: `["body", "files"]`
no upload), `string_too_short` (`question` vazia), `string_too_long`,
`less_than_equal` / `greater_than_equal` (`top_k`), `json_invalid`.

Note que `404` tem `detail` como **string**, e `422` tem `detail` como **lista**.

Falhas que **não** são erros HTTP: arquivos inválidos no upload (vêm em
`results[].error`, com status 200) e falhas do LLM no chat (vêm como evento SSE
`error`, com status 200).
