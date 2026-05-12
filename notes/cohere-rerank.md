# Cohere Rerank: Como o Reranking Melhora a Qualidade do RAG

> **Referência no notebook:** `rerank-rag.ipynb` — implementa a estratégia em dois estágios: `naive_retriever (k=10)` → `CohereRerank (top_n=3)` via `ContextualCompressionRetriever`.

---

## O Problema que o Reranking Resolve

A busca vetorial por similaridade de cosseno é rápida e eficiente, mas tem uma limitação fundamental:

```
Query: "Quais são as penalidades para empresas que violarem a lei de IA?"

Busca vetorial retorna (por similaridade de embedding):
  1. "Art. 36 — infrações serão punidas com advertência..." (score: 0.91) ✅
  2. "Art. 1º — esta lei estabelece normas de IA..." (score: 0.88) ❌ irrelevante
  3. "Art. 15 — empresas devem garantir transparência..." (score: 0.85) ❌ tangencial
  4. "Art. 36 § 1º — multas serão aplicadas conforme..." (score: 0.83) ✅ relevante
  5. "Considerações sobre IA na Europa..." (score: 0.80) ❌ irrelevante
```

O problema: `score 0.88` não significa que o chunk é relevante para **responder** a pergunta — significa apenas que ele é **semanticamente próximo** da pergunta no espaço vetorial.

---

## A Solução em Dois Estágios

```
                        ESTÁGIO 1: Recuperação Ampla
                        ─────────────────────────────
Query ──→ Embedding ──→ Busca Vetorial (k=10) ──→ 10 candidatos
                                                         │
                        ESTÁGIO 2: Reranking Preciso      │
                        ─────────────────────────────     │
                                          CohereRerank ←──┘
                                          (top_n=3)
                                               │
                                        3 chunks REALMENTE
                                        relevantes para o LLM
```

---

## CohereRerank: Bi-Encoder vs Cross-Encoder

### Modelo de Embedding (Bi-Encoder) — usado na busca inicial

```
Query  ──→ Encoder ──→ vetor_query [0.23, -0.89, ...]
Chunk  ──→ Encoder ──→ vetor_chunk [0.25, -0.88, ...]

Score = cosine_similarity(vetor_query, vetor_chunk)
```

- Vetores são calculados **independentemente**
- Muito rápido (vetores ficam pré-calculados)
- Não entende a relação entre query e chunk

### Cross-Encoder (Reranker Cohere) — refinamento

```
[Query + Chunk] ──→ Modelo de Linguagem ──→ Score de Relevância
                     (processado JUNTO)        (0.0 a 1.0)
```

- Query e chunk são processados **em conjunto** pelo modelo
- O modelo entende a relação semântica entre os dois
- Muito mais preciso, mas mais lento (não pode ser pré-calculado)

**Analogia:** O bi-encoder é como comparar fotos de dois rostos. O cross-encoder é como colocar as duas pessoas frente a frente e perguntar "quão compatíveis são?".

---

## Implementação no Notebook

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Estágio 1: Retriever base (amplo)
naive_retriever = vectordb.as_retriever(search_kwargs={'k': 10})

# Estágio 2: Reranker
rerank = CohereRerank(
    model='rerank-multilingual-v3.0',  # Funciona bem em português
    top_n=3                             # Seleciona os 3 melhores de 10
)

# Orquestrador dos dois estágios
compressor_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,       # O que filtra/reordena
    base_retriever=naive_retriever # O que busca inicialmente
)
```

### Por que `k=10` no retriever base?

Quanto mais candidatos o reranker recebe, melhor a qualidade final — mas maior o custo. A escolha de 10 é um equilíbrio:

- **Muito poucos (k=3)**: O reranker não tem candidatos suficientes para escolher
- **Muitos (k=50)**: Custo alto de chamadas à API Cohere, latência maior
- **k=10**: Suficiente para cobrir documentos relevantes, custo razoável

---

## Modelos Cohere de Reranking

| Modelo                        | Idiomas    | Uso ideal                              |
|-------------------------------|------------|----------------------------------------|
| `rerank-english-v3.0`         | Inglês     | Documentos apenas em inglês            |
| `rerank-multilingual-v3.0`    | 100+ línguas | Português, misto, global             |
| `rerank-english-v2.0`         | Inglês     | Versão anterior (menos precisa)        |

O notebook usa `rerank-multilingual-v3.0` pois o documento (projeto de lei) está em **português**.

---

## ContextualCompressionRetriever: O Nome Explica

O nome "compressão contextual" vem da ideia de **comprimir** o conjunto de documentos recuperados para apenas os mais relevantes:

```
10 documentos candidatos ──→ CohereRerank ──→ 3 documentos comprimidos
(amplo, pode ter ruído)         (filtra)      (denso, alta relevância)
```

A "compressão" pode ser de dois tipos:
1. **Seleção**: Retorna apenas os top-N documentos (como no reranking)
2. **Extração**: Retorna apenas os trechos relevantes dentro dos documentos (compressão contextual)

---

## Comparando os Pipelines RAG

```
Simple RAG:
  Query → k=3 chunks (por cosseno) → LLM
  ✅ Simples e rápido
  ❌ Pode trazer chunks superficialmente similares mas irrelevantes

Parent RAG:
  Query → k filhos → pais correspondentes → LLM
  ✅ Contexto mais rico para o LLM
  ❌ Ainda usa similaridade de cosseno para selecionar

Rerank RAG:
  Query → k=10 chunks (por cosseno) → top_n=3 (por cross-encoder) → LLM
  ✅ Melhor qualidade de relevância
  ✅ LLM recebe apenas chunks realmente úteis
  ❌ Latência maior (2 chamadas de API: OpenAI + Cohere)
  ❌ Custo maior
```

---

## O Pipeline LCEL do Rerank RAG

```python
setup_retrival = RunnableParallel({
    'question': RunnablePassthrough(),
    'context': compressor_retriever,  # Executa os 2 estágios
})

output_parser = StrOutputParser()

compressor_retriever_chain = setup_retrival | rag_prompt | llm | output_parser
```

### Fluxo completo ao invocar

```
"Quais os pontos de risco do marco legal de IA?"
    │
    ▼ RunnableParallel
    ├── question: "Quais os pontos de risco..."  (RunnablePassthrough)
    └── context:
          │
          ▼ naive_retriever (k=10)
          [chunk1, chunk2, ..., chunk10]
          │
          ▼ CohereRerank (top_n=3)
          [chunk_mais_relevante, chunk2, chunk3]
    │
    ▼ rag_prompt (formata o template)
    "Você é especialista... Query: ... Context: chunk1 chunk2 chunk3"
    │
    ▼ ChatOpenAI (gpt-3.5-turbo, max_tokens=300)
    AIMessage(content="Os principais pontos de risco são...")
    │
    ▼ StrOutputParser
    "Os principais pontos de risco são..."
```

---

## Quando Usar Reranking

**Use reranking quando:**
- Qualidade da resposta é crítica
- Documentos são longos e heterogêneos (múltiplos tópicos)
- Perguntas são complexas e precisas
- Orçamento de latência é maior (2-3s adicionais são aceitáveis)

**Evite reranking quando:**
- Latência é crítica (aplicações real-time)
- Base de conhecimento é pequena e homogênea
- Custo de API é limitante
- Respostas já são satisfatórias com busca simples
