# Parent Document Retriever: Estratégia Filho-Pai

> **Referência no notebook:** `parent-rag.ipynb` — implementa o `ParentDocumentRetriever` com `child_splitter` (200 chars) e `parent_splitter` (4000 chars).

---

## O Problema Fundamental do RAG Simples

No RAG simples (`simple-rag-documents.ipynb`), existe um trade-off difícil:

```
┌──────────────────────────────────────────────────────────┐
│                 O DILEMA DO CHUNK SIZE                   │
│                                                          │
│  Chunks PEQUENOS:                                        │
│  ✅ Embeddings precisos (menos ruído semântico)          │
│  ❌ Pouco contexto para o LLM gerar uma boa resposta     │
│                                                          │
│  Chunks GRANDES:                                         │
│  ✅ Contexto rico para o LLM                             │
│  ❌ Embeddings imprecisos (mistura muitos tópicos)       │
└──────────────────────────────────────────────────────────┘
```

**Exemplo prático:**

Imagine um parágrafo de 4000 caracteres que fala sobre "fundamentos de IA", "regulação europeia" e "sanções administrativas". O embedding desse chunk vai ser uma média de todos esses tópicos, ficando "diluído" — ele vai ser medianamente similar a muitas perguntas, mas muito preciso para nenhuma específica.

---

## A Solução: Dois Documentos Para o Preço de Um

O `ParentDocumentRetriever` resolve o trade-off com uma estratégia elegante:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESTRATÉGIA PARENT-CHILD                      │
│                                                                 │
│  INDEXAÇÃO (offline):                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Documento Original (página inteira)                      │   │
│  │                                                          │   │
│  │  [chunk pai A - 4000 chars] [chunk pai B - 4000 chars]  │   │
│  │      │                            │                      │   │
│  │   [f1][f2][f3]                 [f4][f5][f6]             │   │
│  │   200c 200c 200c               200c 200c 200c           │   │
│  │   filhos                       filhos                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  BUSCA (online):                                               │
│  Query → Embedding → Busca filhos (precisão) → Retorna PAIS   │
└─────────────────────────────────────────────────────────────────┘
```

- **Filhos** (200 chars): Servem apenas para a busca semântica (não vão para o LLM)
- **Pais** (4000 chars): São enviados ao LLM com contexto completo

---

## Dois Storages com Responsabilidades Distintas

```python
# Storage 1: Para os embeddings dos filhos (busca vetorial)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory='childVectorDB'  # Persiste em disco
)

# Storage 2: Para os documentos pais (recuperação por ID)
store = InMemoryStore()  # Em memória RAM, sem persistência
```

### Por que InMemoryStore para os pais?

O `InMemoryStore` funciona como um dicionário Python simples:

```python
# Internamente é equivalente a:
store = {}  # {doc_id: documento_pai}

# Ao adicionar documentos:
store["uuid-1234"] = Document(page_content="texto longo de 4000 chars...")
store["uuid-5678"] = Document(page_content="outro texto longo...")

# Na busca:
pai = store["uuid-1234"]  # Acesso por chave, O(1)
```

Não precisa de busca vetorial — o retriever já sabe o ID do pai após encontrar o filho.

### Por que Chroma persiste mas InMemoryStore não?

- **Chroma (filhos)**: Gerar embeddings é caro (chamadas à API OpenAI). Faz sentido salvar em disco para reutilizar.
- **InMemoryStore (pais)**: É só texto. Recarregar do PDF é rápido. Persistência adicionaria complexidade sem ganho real.

> **Atenção:** Por isso, ao reiniciar o notebook, é necessário executar `add_documents` novamente para popular o `InMemoryStore`.

---

## Como o ParentDocumentRetriever Funciona Internamente

```python
parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,       # Onde buscar (filhos)
    docstore=store,                # De onde recuperar (pais)
    child_splitter=child_splitter, # Como dividir para indexar (200 chars)
    parent_splitter=parent_splitter # Como dividir para armazenar (4000 chars)
)

parent_document_retriever.add_documents(pages, ids=None)
```

### O que acontece em `add_documents`:

```
páginas do PDF (31 docs)
        ↓ parent_splitter (chunk_size=4000)
documentos pais (~31-50 pais)
        │                     ↓ armazena no InMemoryStore com UUID
        │                     {"uuid-a": Doc(4000 chars), "uuid-b": Doc(4000 chars)}
        │
        ↓ child_splitter (chunk_size=200)
chunks filhos (2373 filhos!)
        │
        ↓ gera embeddings + referencia doc_id do pai
        ↓ armazena no Chroma
        {"uuid-filho-1": embedding + metadata{doc_id: "uuid-a"}}
        {"uuid-filho-2": embedding + metadata{doc_id: "uuid-a"}}
        ...
```

### O que acontece na busca (`.invoke(query)`):

```python
query = "Quais os principais riscos do marco legal de IA?"
docs = parent_document_retriever.invoke(query)

# Internamente:
# 1. query → embedding
# 2. Busca filhos similares no Chroma (top-k filhos)
# 3. Lê doc_id do metadado de cada filho
# 4. Busca pais no InMemoryStore pelo doc_id
# 5. Deduplica pais (filhos do mesmo pai retornam o pai uma só vez)
# 6. Retorna lista de documentos PAIS (4000 chars cada)
```

---

## Impacto Real: 31 Páginas → 2373 Chunks Filhos

```python
# O notebook mostra esse número:
data = parent_document_retriever.vectorstore.get()
# 2373 chunks filhos para 31 páginas

# Cálculo aproximado:
# 31 páginas × ~200 chars/linha × ~20 linhas/página = ~124.000 chars total
# 124.000 / 200 chars por filho = ~620 filhos esperados
# 2373 >> 620 — por quê?

# Resposta: o parent_splitter também divide as páginas em pais menores
# Cada página (~4000 chars) pode virar 1-3 pais
# Cada pai de ~4000 chars → ~20 filhos de 200 chars
# 31 páginas → ~50-80 pais → ~1000-1600 filhos (mais overlap = mais filhos)
```

---

## Comparação de Números: Simple RAG vs Parent RAG

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│                    │    Simple RAG        │    Parent RAG        │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Chunks para busca  │ 31 (1 por página)    │ 2373 (filhos 200c)  │
│ Enviado ao LLM     │ Top-3 chunks (4000c) │ Top-N pais (4000c)  │
│ Precisão busca     │ ★★★☆☆               │ ★★★★★               │
│ Contexto para LLM  │ ★★★☆☆               │ ★★★★★               │
│ Memória usada      │ Baixa               │ Alta (2373 embeddings)│
│ Custo de indexação │ Baixo               │ Alto (mais embeddings)│
└────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Quando Usar Parent Document Retriever

**Use quando:**
- Documentos longos com tópicos variados por seção
- Perguntas precisas que precisam de contexto amplo para ser respondidas
- Você quer a precisão de chunks pequenos mas a riqueza de chunks grandes

**Evite quando:**
- Orçamento de tokens é crítico (pais grandes consomem mais tokens)
- Documentos são curtos e coesos (não há ganho)
- Latência é prioridade (mais complexidade = mais tempo)

---

## Alternativa: Small-to-Big sem ParentDocumentRetriever

É possível implementar a mesma estratégia manualmente:

```python
# Versão manual simplificada
child_to_parent = {}  # Mapeamento filho → pai

for i, parent_chunk in enumerate(parent_chunks):
    for child_chunk in split_into_children(parent_chunk):
        child_id = str(uuid.uuid4())
        child_to_parent[child_id] = parent_chunk
        vectorstore.add_texts(
            texts=[child_chunk.page_content],
            ids=[child_id]
        )

# Na busca:
child_results = vectorstore.similarity_search(query, k=5)
parent_docs = [child_to_parent[child.metadata['id']] for child in child_results]
```

O `ParentDocumentRetriever` automatiza exatamente esse padrão.
