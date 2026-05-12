# RAGs com LangChain

Repositório de estudo desenvolvido durante a trilha de IA para devs da **Rocketseat**, explorando diferentes arquiteturas de **Retrieval-Augmented Generation (RAG)** com LangChain e OpenAI.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Variáveis de Ambiente](#variáveis-de-ambiente)
- [Notebooks](#notebooks)
  - [simple-rag-documents.ipynb](#1-simple-rag-documentsipynb)
  - [parent-rag.ipynb](#2-parent-ragipynb)
  - [code-review-rag.ipynb](#3-code-review-ragipynb)
  - [rerank-rag.ipynb](#4-rerank-ragipynb)
- [Comparativo das Arquiteturas](#comparativo-das-arquiteturas)
- [Notas Complementares](#notas-complementares)
  - [rag-concepts.md — Chunk Size e Overlap](#rag-conceptsmd--chunk-size-e-overlap)
  - [chunk-to-vector.md — De Chunk para Vetor](#chunk-to-vectormd--de-chunk-para-vetor)
  - [estrategias-avancadas-rag.md — Estratégias Avançadas](#estrategias-avancadas-ragmd--estratégias-avançadas)
  - [rag-ranking.md — Busca e Ranking](#rag-rankingmd--busca-e-ranking)
  - [rag-vs-code-review.md — Simple RAG vs Code Review RAG](#rag-vs-code-reviewmd--simple-rag-vs-code-review-rag)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

Este projeto implementa **4 variações de pipelines RAG**, cada uma explorando uma estratégia diferente de recuperação e geração. O documento base utilizado é um **Projeto de Lei sobre Inteligência Artificial** (31 páginas), exceto no notebook de Code Review, que analisa o repositório oficial do LangChain.

```
Documento / Código  →  Chunks  →  Embeddings  →  Vector Store
                                                      ↕
Pergunta do usuário  →  Retriever  →  Contexto  →  LLM  →  Resposta
```

---

## Pré-requisitos

- [Anaconda](https://www.anaconda.com/) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.12
- Chave de API da [OpenAI](https://platform.openai.com/)
- Chave de API da [Cohere](https://cohere.com/) _(necessária apenas para o `rerank-rag.ipynb`)_

---

## Instalação

Clone o repositório e crie o ambiente conda com todas as dependências:

```bash
git clone <url-do-repositorio>
cd rocket-ia-rags

# Criar e ativar o ambiente
conda env create -f environment.yml
conda activate rag-application

# Instalar o kernel do Jupyter para o ambiente
python -m ipykernel install --user --name rag-application
```

---

## Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto baseado no `.env.example`:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...       # Necessária para todos os notebooks
COHERE_API_KEY=...          # Necessária apenas para rerank-rag.ipynb
```

---

## Notebooks

### 1. [simple-rag-documents.ipynb](./simple-rag-documents.ipynb)

**RAG Simples com Documentos PDF**

O ponto de partida do projeto. Implementa o pipeline RAG clássico (Retrieve → Augment → Generate) sobre um PDF de projeto de lei sobre IA.

**Pipeline:**

```
PDF  →  PyPDFLoader  →  RecursiveCharacterTextSplitter  →  Chroma (persist)
                                                              ↕
Pergunta  →  Retriever (k=3)  →  load_qa_chain (stuff)  →  GPT-3.5-turbo  →  Resposta
```

**Principais componentes:**
| Componente | Configuração |
|---|---|
| Loader | `PyPDFLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` — `chunk_size=4000`, `chunk_overlap=20` |
| Embeddings | `OpenAIEmbeddings` (`text-embedding-3-small`) |
| Vector Store | `Chroma` — persistência em `./text_index` |
| Retriever | Similaridade de cosseno, `k=3` |
| Chain | `load_qa_chain` — tipo `stuff` |
| LLM | `gpt-3.5-turbo`, `max_tokens=200` |

**Destaques:**

- Armazenamento **persistente** do vector store (não recalcula a cada execução)
- Exposição dos metadados dos chunks recuperados (página, fonte, índice)
- Função `ask()` que encapsula o fluxo RAG completo

---

### 2. [parent-rag.ipynb](./parent-rag.ipynb)

**Parent Document RAG — Estratégia Parent-Child**

Resolve o trade-off do RAG simples: chunks pequenos geram embeddings precisos, mas têm pouco contexto para o LLM. Chunks grandes têm mais contexto, mas embeddings menos precisos. A solução: usar **dois tamanhos de chunk com propósitos distintos**.

**Pipeline:**

```
PDF  →  child_splitter (200 chars)  →  Chroma (embeddings para busca)
     →  parent_splitter (4000 chars)  →  InMemoryStore (contexto para LLM)
                                              ↕
Pergunta  →  ParentDocumentRetriever  →  Documentos PAI  →  GPT-3.5-turbo  →  Resposta
```

**Principais componentes:**
| Componente | Configuração |
|---|---|
| Loader | `PyPDFLoader` |
| Child Splitter | `RecursiveCharacterTextSplitter` — `chunk_size=200` |
| Parent Splitter | `RecursiveCharacterTextSplitter` — `chunk_size=4000`, `chunk_overlap=200` |
| Embeddings | `OpenAIEmbeddings` (`text-embedding-3-small`) |
| Vector Store (filhos) | `Chroma` — persistência em `./childVectorDB` |
| Doc Store (pais) | `InMemoryStore` — em memória RAM |
| Retriever | `ParentDocumentRetriever` |
| LLM | `gpt-3.5-turbo`, `max_tokens=500` |
| Chain | LCEL com `RunnableParallel` e `StrOutputParser` |

**Destaques:**

- **2373 chunks filhos** indexados no Chroma
- Busca por chunks pequenos → retorna o documento pai correspondente ao LLM
- Pipeline montado com sintaxe moderna LCEL: `setup_retrival | rag_prompt | llm | output_parser`
- Visualização dos chunks no vector store com `pandas.DataFrame`

---

### 3. [code-review-rag.ipynb](./code-review-rag.ipynb)

**Code Review RAG — Análise de Repositório Python**

Aplica RAG para **revisão de código**, usando como fonte o repositório oficial do LangChain. O parser de linguagem divide o código por unidades semânticas (funções, classes), não apenas por caracteres.

**Pipeline:**

```
GitHub Repo (clone)  →  GenericLoader + LanguageParser  →  641 documentos Python
                     →  RecursiveCharacterTextSplitter.from_language(PYTHON)  →  1803 chunks
                     →  Chroma (MMR retriever, k=8)
                                    ↕
Pergunta  →  MMR Retriever  →  create_stuff_documents_chain  →  create_retrieval_chain  →  Resposta
```

**Principais componentes:**
| Componente | Configuração |
|---|---|
| Loader | `GenericLoader` + `LanguageParser(Language.PYTHON, parser_threshold=500)` |
| Text Splitter | `RecursiveCharacterTextSplitter.from_language(PYTHON)` — `chunk_size=2000`, `chunk_overlap=200` |
| Embeddings | `OpenAIEmbeddings(disallowed_special=())` |
| Vector Store | `Chroma` — em memória |
| Retriever | `MMR` (`search_type='mmr'`, `k=8`) |
| Chain | `create_stuff_documents_chain` + `create_retrieval_chain` |
| LLM | `gpt-3.5-turbo`, `max_tokens=1000` |

**Destaques:**

- **Clonagem automática** do repositório LangChain via `GitPython`
- **MMR (Maximal Marginal Relevance)**: busca documentos relevantes E diversos, evitando redundância
- Prompt estruturado com roles `system`/`user` para análise em 4 categorias: Bugs, Performance, Qualidade e Python idiomático
- Análise de código **real** do repositório (não apenas conhecimento treinado do modelo)

---

### 4. [rerank-rag.ipynb](./rerank-rag.ipynb)

**Rerank RAG — Recuperação em Dois Estágios**

Introduz uma etapa de **reranking** com o modelo Cohere para filtrar e reordenar os documentos recuperados por relevância real, não apenas por similaridade de embedding.

**Pipeline:**

```
PDF  →  RecursiveCharacterTextSplitter  →  Chroma (naiveDB)
                                              ↕
Pergunta  →  Naive Retriever (k=10)  →  CohereRerank (top_n=3)  →  GPT-3.5-turbo  →  Resposta
              (similaridade vetorial)    (relevância semântica real)
```

**Principais componentes:**
| Componente | Configuração |
|---|---|
| Loader | `PyPDFLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` — `chunk_size=4000`, `chunk_overlap=20` |
| Embeddings | `OpenAIEmbeddings` (`text-embedding-3-small`) |
| Vector Store | `Chroma` — persistência em `./naiveDB` |
| Naive Retriever | Similaridade de cosseno, `k=10` |
| Reranker | `CohereRerank` — `model='rerank-multilingual-v3.0'`, `top_n=3` |
| Retriever Final | `ContextualCompressionRetriever` |
| LLM | `gpt-3.5-turbo`, `max_tokens=300` |
| Chain | LCEL com `RunnableParallel` e `StrOutputParser` |

**Destaques:**

- Estratégia em **dois estágios**: busca ampla (k=10) + filtragem precisa (top_n=3)
- `CohereRerank` avalia cada par (pergunta, chunk) e atribui um score de relevância real
- Modelo multilingue — funciona bem com perguntas e documentos em português
- Requer `COHERE_API_KEY` no `.env`

---

## Notas Complementares

A pasta [`notes/`](./notes/) contém documentação de apoio sobre os conceitos teóricos aplicados nos notebooks. Cada arquivo aprofunda um tema específico do pipeline RAG.

---

### [rag-concepts.md](./notes/rag-concepts.md) — Chunk Size e Overlap

**Conceito central:** Como a divisão de texto em chunks impacta a qualidade do RAG.

Aborda os três parâmetros fundamentais de qualquer pipeline:

| Conceito          | Descrição                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| **Chunk Size**    | Limite de caracteres/tokens por bloco — pequeno demais perde contexto, grande demais traz ruído  |
| **Chunk Overlap** | Sobreposição entre blocos adjacentes — garante que frases cortadas na divisão não percam sentido |
| **Token**         | Unidade básica dos LLMs — ~750 palavras por 1.000 tokens; determina custo e limite de contexto   |
| **Granularidade** | Equilíbrio entre precisão de busca (chunks pequenos) e riqueza de contexto (chunks grandes)      |

> 📌 **Relevante para:** todos os notebooks — especialmente a escolha de `chunk_size=4000` no `simple-rag-documents.ipynb` vs. `chunk_size=200` (filho) no `parent-rag.ipynb`.

---

### [chunk-to-vector.md](./notes/chunk-to-vector.md) — De Chunk para Vetor

**Conceito central:** Como texto é transformado em vetores numéricos internamente.

Explica o processo completo de embedding, do texto ao vetor:

```
Texto → Tokenização → Embedding Layer → Self-Attention (Transformer) → Vetor final (1536 dims)
```

Tópicos abordados:

- **Tokenização** — subword tokenization, mapeamento para IDs numéricos
- **Embedding Layer** — tabela de pesos treinados que converte IDs em vetores
- **Self-Attention** — como o Transformer contextualiza cada token com os demais
- **Tratamento de números** — por que embeddings falham com valores numéricos específicos e como mitigar
- **Modelos comparados** — `text-embedding-3-small` (512 dims), `text-embedding-3-large` (3072 dims), `Sentence-BERT`, `ColBERT`

> 📌 **Relevante para:** entender por que `OpenAIEmbeddings(model='text-embedding-3-small')` é usado em todos os notebooks e o que acontece ao chamar `Chroma.from_documents()`.

---

### [estrategias-avancadas-rag.md](./notes/estrategias-avancadas-rag.md) — Estratégias Avançadas

**Conceito central:** Técnicas além do RAG simples para melhorar precisão e reduzir alucinações.

Cobre 7 estratégias do estado da arte:

| Estratégia                     | Problema que resolve                                                    | Implementada neste repo             |
| ------------------------------ | ----------------------------------------------------------------------- | ----------------------------------- |
| **Re-ranking**                 | Similaridade vetorial ≠ relevância real                                 | ✅ `rerank-rag.ipynb`               |
| **Multi-Query Retrieval**      | Perguntas ambíguas ou com termos diferentes                             | ❌                                  |
| **HyDE**                       | Buscar "pergunta vs resposta" é menos eficaz que "resposta vs resposta" | ❌                                  |
| **Contextual Compression**     | Chunks com informação irrelevante desperdiçam contexto                  | ✅ `ContextualCompressionRetriever` |
| **Self-RAG / Self-Reflection** | LLM responde mesmo sem ter a informação                                 | ❌                                  |
| **GraphRAG**                   | Busca vetorial não conecta informações entre documentos distintos       | ❌                                  |
| **Agentic RAG**                | Uma única estratégia não serve para todos os tipos de pergunta          | ❌                                  |

> 📌 **Relevante para:** `rerank-rag.ipynb` implementa Re-ranking + Contextual Compression; as demais estratégias representam possíveis evoluções do projeto.

---

### [rag-ranking.md](./notes/rag-ranking.md) — Busca e Ranking

**Conceito central:** Como documentos são recuperados e ordenados por relevância.

Diagrama do pipeline completo:

```
Pergunta → Embedding da Query → Busca (candidatos) → Ranking → Reranking → Top-K → LLM
```

Estratégias de busca comparadas:

| Estratégia                      | Velocidade | Qualidade  | Caso de uso            |
| ------------------------------- | ---------- | ---------- | ---------------------- |
| **Embedding puro** (cosseno)    | ⚡⚡⚡     | ⭐⭐⭐     | Geral, base grande     |
| **BM25** (keyword)              | ⚡⚡⚡     | ⭐⭐       | Palavras-chave claras  |
| **Híbrido** (embedding + BM25)  | ⚡⚡       | ⭐⭐⭐⭐   | Produção (recomendado) |
| **+ Reranking** (Cross-Encoder) | ⚡         | ⭐⭐⭐⭐⭐ | Qualidade crítica      |

Também documenta problemas comuns:

- **Lost in the Middle** — LLM ignora informações no meio da lista de chunks
- **Chunks fragmentados** — corte no meio de uma ideia quebra o contexto
- **Semantic Drift** — documentos matematicamente similares mas semanticamente irrelevantes

E comparativo de bancos de dados vetoriais (Weaviate, Pinecone, Milvus, Qdrant, Chroma, FAISS).

> 📌 **Relevante para:** `rerank-rag.ipynb` (estratégia de busca em dois estágios) e `code-review-rag.ipynb` (busca MMR para diversidade).

---

### [rag-vs-code-review.md](./notes/rag-vs-code-review.md) — Simple RAG vs Code Review RAG

**Conceito central:** Por que processar código exige uma arquitetura diferente de processar texto.

| Característica | Simple RAG                 | Code Review RAG                                 |
| -------------- | -------------------------- | ----------------------------------------------- |
| **Loader**     | `PyPDFLoader` (texto puro) | `GenericLoader` + `LanguageParser` (estrutural) |
| **Splitter**   | Caracteres genéricos       | Sintaxe Python (`from_language`)                |
| **Busca**      | Similaridade simples       | MMR (relevância + diversidade)                  |
| **Abordagem**  | Funções manuais (`ask()`)  | Chains orquestradas (`create_retrieval_chain`)  |
| **Foco**       | Recuperação de informação  | Análise lógica e contextual                     |

Destaques:

- **LanguageParser** entende a sintaxe Python — não corta funções ou classes no meio
- **MMR** garante que a IA veja diferentes partes do sistema (definição, método, utilitários), não 8 variações do mesmo padrão
- **`create_retrieval_chain`** encapsula automaticamente todo o pipeline (retrieval → formatação → LLM → resposta)

> 📌 **Relevante para:** justifica as escolhas arquiteturais do `code-review-rag.ipynb` em relação ao `simple-rag-documents.ipynb`.

---

## Estrutura do Projeto

```
rocket-ia-rags/
│
├── 📓 simple-rag-documents.ipynb   # RAG clássico com PDF
├── 📓 parent-rag.ipynb             # Parent Document RAG
├── 📓 code-review-rag.ipynb        # Code Review com repositório Git
├── 📓 rerank-rag.ipynb             # RAG com reranking Cohere
│
├── assets/
│   ├── anexo-projeto-de-lei.pdf    # Documento base (PDF de 31 páginas)
│   └── presentation-lesson.pdf    # Slides da aula
│
├── notes/                          # Notas de estudo
│   ├── rag-concepts.md
│   ├── chunk-to-vector.md
│   ├── estrategias-avancadas-rag.md
│   ├── rag-ranking.md
│   └── rag-vs-code-review.md
│
├── text_index/                     # Vector Store do simple-rag (Chroma)
├── childVectorDB/                  # Vector Store do parent-rag (Chroma, chunks filhos)
├── naiveDB/                        # Vector Store do rerank-rag (Chroma)
│
├── environment.yml                 # Dependências do ambiente conda
├── .env.example                    # Template das variáveis de ambiente
└── .gitignore
```
