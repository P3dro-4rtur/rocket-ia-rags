# Vector Stores: Chroma e a Persistência de Embeddings

> **Referência nos notebooks:** Todos os 4 notebooks usam `Chroma` como vector store, com `persist_directory` diferentes para cada estratégia.

---

## O Que é um Vector Store?

Um Vector Store é um banco de dados especializado em armazenar e buscar **vetores de alta dimensão** (embeddings) de forma eficiente. Ao contrário de um banco de dados relacional que busca por valores exatos, um vector store busca por **proximidade matemática** no espaço vetorial.

```
Banco Relacional:
  SELECT * FROM docs WHERE content LIKE '%temperatura%'
  → Busca por palavras-chave exatas

Vector Store:
  vectordb.similarity_search("como armazenar medicamentos")
  → Busca por SIGNIFICADO, não por palavras
  → Retorna "temperatura ideal para armazenamento de fármacos"
     mesmo sem ter a palavra "armazenar" no texto
```

---

## Como o Chroma Funciona Internamente

O Chroma usa o algoritmo **HNSW (Hierarchical Navigable Small Worlds)** para busca eficiente:

```
Estrutura do Chroma em disco (persist_directory):
├── chroma.sqlite3       ← Metadados (source, page, autor, etc.)
├── [uuid]/
│   ├── data_level0.bin  ← Vetores de embedding (formato binário)
│   ├── index_metadata   ← Índice para busca HNSW
│   └── header.bin       ← Configurações do índice
```

### O que fica armazenado em cada item:

```python
# Cada documento indexado no Chroma contém:
{
    "id": "uuid-gerado-automaticamente",
    "document": "texto do chunk",           # page_content
    "embedding": [0.234, -0.891, ...],     # vetor de 1536 dimensões
    "metadata": {
        "source": "./assets/doc.pdf",
        "page": 5,
        "page_label": "6",
        "start_index": 1200,
        # ... outros metadados do PyPDFLoader
    }
}
```

---

## Chroma em Cada Notebook

### `simple-rag-documents.ipynb` — Criação e persistência

```python
# Cria o vector store E indexa os chunks em uma única chamada
db = Chroma.from_documents(
    chunks,                    # Lista de Documents (chunks do PDF)
    embedding=embedding_model, # OpenAIEmbeddings → 1536 dimensões
    persist_directory="text_index"  # Salva em disco
)

# Mais tarde, para reutilizar sem reprocessar:
vectordb = Chroma(
    persist_directory='text_index',
    embedding_function=embedding_model
)
```

**Diferença entre `from_documents` e o construtor:**
- `Chroma.from_documents(...)` → Cria novo + indexa (usa API OpenAI → tem custo)
- `Chroma(persist_directory=...)` → Carrega existente (sem custo adicional)

### `parent-rag.ipynb` — Vector store dos filhos

```python
# Cria empty Chroma, os documentos são adicionados pelo ParentDocumentRetriever
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory='childVectorDB'
)

# Os filhos são indexados automaticamente quando:
parent_document_retriever.add_documents(pages, ids=None)
# → divide em filhos (200 chars)
# → gera embeddings de cada filho
# → armazena no Chroma com metadado doc_id apontando para o pai
```

### `rerank-rag.ipynb` — Retriever amplo para reranking

```python
# Cria vector store separado para o rerank pipeline
vectordb = Chroma(
    embedding_function=embeddings_model,
    persist_directory='naiveDB'   # Diretório diferente!
)

# Retriever com k=10 (amplo para dar opções ao reranker)
naive_retriever = vectordb.as_retriever(search_kwargs={'k': 10})
```

### `code-review-rag.ipynb` — Sem persistência + MMR

```python
# Cria diretamente a partir dos chunks (sem persist_directory)
# Isso cria um Chroma em memória — é destruído ao finalizar
db = Chroma.from_documents(
    texts,
    OpenAIEmbeddings(disallowed_special=())  # Necessário para código-fonte
)

# Retriever com MMR em vez de similaridade simples
retriever = db.as_retriever(
    search_type='mmr',         # Maximal Marginal Relevance
    search_kwargs={'k': 8},
)
```

---

## Tipos de Busca no Chroma

### 1. Similaridade de Cosseno (padrão)

```python
retriever = vectordb.as_retriever(search_kwargs={'k': 3})
# Retorna os k documentos com maior similaridade de cosseno
```

**Fórmula:**
```
similarity = cos(θ) = (A · B) / (|A| × |B|)
```
- Resultado entre -1 e 1 (na prática, entre 0 e 1 para embeddings)
- 1.0 = idênticos | 0.0 = sem relação semântica

### 2. MMR — Maximal Marginal Relevance

```python
retriever = db.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 8, 'fetch_k': 20}
)
```

O MMR resolve o problema de resultados redundantes:

```
SEM MMR:
  Query: "funções de Runnable no LangChain"
  
  Resultado 1: "class Runnable: def invoke(...)" - score 0.95
  Resultado 2: "class Runnable: def invoke(...)" - score 0.94  ← DUPLICADO!
  Resultado 3: "class Runnable: def invoke(...)" - score 0.93  ← DUPLICADO!

COM MMR (λ = 0.5):
  Resultado 1: "class Runnable: def invoke(...)" - mais relevante
  Resultado 2: "class RunnableParallel..." - relevante E diferente do 1
  Resultado 3: "class RunnableSequence..." - relevante E diferente dos anteriores
```

**Fórmula MMR:**
```
MMR(Di) = λ × Sim(Di, Query) - (1-λ) × max[Sim(Di, Dj)]
           relevância            diversidade em relação aos selecionados
```

- `λ = 1.0`: Puro relevância (igual a similarity search)
- `λ = 0.0`: Pura diversidade
- `λ = 0.5`: Equilíbrio (padrão do Chroma)

### 3. Similarity Score Threshold

```python
retriever = vectordb.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.7, 'k': 10}
)
# Retorna apenas documentos com score >= 0.7
# Evita trazer documentos pouco relevantes quando k não tem documentos suficientes
```

---

## `disallowed_special=()` no Code Review

```python
# Por que isso é necessário no code-review-rag.ipynb?
db = Chroma.from_documents(
    texts,
    OpenAIEmbeddings(disallowed_special=())  # <-- Este parâmetro
)
```

O tokenizador `tiktoken` (usado pela API OpenAI) por padrão **rejeita** tokens especiais como `<|endoftext|>`, `<|fim_prefix|>`, etc.

Arquivos de código-fonte Python frequentemente contêm strings como essas em exemplos de código, docstrings de testes, etc. Sem `disallowed_special=()`, o processo de embedding **lança uma exceção** ao processar esses arquivos.

`disallowed_special=()` define que **nenhum token especial é proibido** — todos são tratados como texto normal.

---

## Persistência vs. Re-indexação

```
┌─────────────────────────────────────────────────────────┐
│           QUANDO FAZER CADA COISA                       │
│                                                         │
│  from_documents() → RE-INDEXA TUDO                     │
│  ├── Chama API OpenAI (custo de tokens)                │
│  ├── Demora (1-5 minutos para documentos médios)       │
│  └── Use apenas quando o documento MUDAR               │
│                                                         │
│  Chroma(persist_directory=...) → CARREGA DO DISCO      │
│  ├── Sem custo de API                                  │
│  ├── Instantâneo                                       │
│  └── Use sempre que o documento não mudou              │
└─────────────────────────────────────────────────────────┘
```

**Padrão recomendado:**

```python
import os

PERSIST_DIR = "text_index"

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # Carrega existente (sem custo)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
    print("✅ Vector store carregado do disco")
else:
    # Cria novo (com custo de API)
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )
    print("✅ Vector store criado e salvo em disco")
```

---

## Múltiplos Diretórios nos Notebooks

Cada notebook usa um diretório diferente para evitar conflitos:

| Notebook               | persist_directory | Estratégia         |
|------------------------|-------------------|--------------------|
| `simple-rag-documents` | `text_index/`     | Chunks de 4000c   |
| `parent-rag`           | `childVectorDB/`  | Chunks filhos 200c |
| `rerank-rag`           | `naiveDB/`        | Chunks de 4000c    |
| `code-review-rag`      | *(em memória)*    | Sem persistência   |

> **Por que separados?** Cada RAG usa embeddings diferentes (modelo, chunk size). Misturar indexações diferentes no mesmo diretório corromperia os resultados.
