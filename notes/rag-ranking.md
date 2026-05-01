# Busca e Ranking em RAG

## O Pipeline Completo

```
Pergunta do Usuário: "Qual é a temperatura recomendada para armazenar medicamentos?"
        ↓
[1. Embedding da Query]
        ↓
[2. Busca nos Chunks] ← Recupera candidatos
        ↓
[3. Ranking] ← Ordena por relevância
        ↓
[4. Reordenação (opcional)] ← Refina
        ↓
Top-K chunks → LLM → Resposta
```

---

## 1️⃣ EMBEDDING DA QUERY

Usa o **mesmo modelo de embedding dos chunks**:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

query = "Qual é a temperatura recomendada para armazenar medicamentos?"
query_embedding = model.encode(query)  # [0.234, -0.891, 0.456, ...]
# Dimensão: 384

# Chunks já têm embeddings salvos no banco de dados:
# chunk_1: [0.245, -0.880, 0.470, ...] (384-dim)
# chunk_2: [0.100, -0.500, 0.200, ...] (384-dim)
# ...
```

---

## 2️⃣ BUSCA (Retrieval)

Existem várias estratégias para encontrar candidatos:

### **A. Busca Vetorial (Similarity Search)**

Calcula **similaridade de cosseno** entre query e chunks:

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Produto escalar normalizado"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

query_emb = [0.234, -0.891, 0.456]
chunk_1 = [0.245, -0.880, 0.470]
chunk_2 = [0.100, -0.500, 0.200]

score_1 = cosine_similarity(query_emb, chunk_1)  # 0.997 (MUITO similar!)
score_2 = cosine_similarity(query_emb, chunk_2)  # 0.812 (menos similar)
```

**Por que cosseno?**

- Mede **ângulo** entre vetores, não magnitude
- Dois documentos com conteúdo igual mas tamanhos diferentes terão score 1.0
- Escala de 0 a 1 (intuitivo)

### **B. Busca por Palavras-chave (BM25)**

Busca tradicional baseada em frequência (tipo Google antigo):

```python
from rank_bm25 import BM25Okapi

documents = [
    "Medicamentos devem ser armazenados entre 15 e 25°C",
    "A insulina requer armazenamento em geladeira",
    "Temperatura ambiente é adequada para comprimidos"
]

# Tokenizar
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

query = "temperatura armazenar medicamentos"
scores = bm25.get_scores(query.split())
# [0.45, 0.12, 0.38]  ← Doc 1 é mais relevante
```

**Vantagens**:

- Rápido
- Bom para palavras-chave específicas
- Determinístico (sem rede neural)

### **C. Busca Híbrida (Vetorial + BM25)**

Combina ambas as abordagens:

```python
def hybrid_search(query, chunks, alpha=0.5):
    """
    alpha: peso do embedding (0-1)
    1-alpha: peso do BM25
    """

    # Score vetorial (normalizado 0-1)
    query_emb = model.encode(query)
    vector_scores = [
        cosine_similarity(query_emb, chunk['embedding'])
        for chunk in chunks
    ]
    vector_scores = normalize(vector_scores)  # 0-1

    # Score BM25
    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = normalize(bm25_scores)  # 0-1

    # Combinação ponderada
    hybrid_scores = [
        alpha * v + (1 - alpha) * b
        for v, b in zip(vector_scores, bm25_scores)
    ]

    return sorted(zip(chunks, hybrid_scores),
                  key=lambda x: x[1], reverse=True)

# Exemplo com alpha=0.6 (70% embedding, 30% BM25)
top_chunks = hybrid_search(query, chunks, alpha=0.7)
```

---

## 3️⃣ RANKING

Depois de recuperar candidatos, é preciso **ordená-los por relevância**:

### **Métrica: Similaridade de Cosseno**

```
query_embedding:  [0.5, -0.3, 0.8, ...]
chunk_embedding:  [0.4, -0.2, 0.9, ...]

similaridade = 0.5×0.4 + (-0.3)×(-0.2) + 0.8×0.9 + ...
            = 0.2 + 0.06 + 0.72 + ...
            ≈ 0.98  (Muito relevante!)
```

### **Ranking em Prática**

```python
# Recupera todos os chunks da base (ou top-1000)
candidates = vector_db.search(query_embedding, top_k=1000)

# Já vêm ordenados por score!
for i, (chunk, score) in enumerate(candidates[:5]):
    print(f"{i+1}. Score: {score:.4f}")
    print(f"   {chunk[:100]}...")
    print()

# Output:
# 1. Score: 0.9847
#    Medicamentos devem ser armazenados entre 15 e 25°C para...
#
# 2. Score: 0.9234
#    A temperatura de armazenamento é crítica para a...
#
# 3. Score: 0.8912
#    Refrigeração é necessária para insulina e...
```

---

## 4️⃣ RERANKING (Refinamento)

**Problema**: Top chunks similares podem não ser os mais relevantes!

```
Query: "medicamentos para dengue"

Top-5 por embedding:
1. "A dengue é transmitida por mosquito" (0.92)  ← Irrelevante!
2. "Paracetamol é medicamento antipirético" (0.88)
3. "Ibuprofeno trata dor e febre da dengue" (0.85)
4. ...
```

### **Solução: Reranking com Cross-Encoder**

Um **cross-encoder** lê a query E o chunk juntos:

```python
from sentence_transformers import CrossEncoder

# Modelo que entende pares (query, documento)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

query = "medicamentos para dengue"
chunks = [chunk1, chunk2, chunk3, ...]

# Calcula score de cada par (query, chunk)
pairs = [[query, chunk] for chunk in chunks]
scores = reranker.predict(pairs)
# [0.45, 0.89, 0.92, ...]

# Reordena
ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
```

**Diferença**:

- **Siamese (embedding)**: `similarity(embedding_query, embedding_chunk)`
- **Cross-Encoder**: `score_network(query, chunk)` (processa par inteiro)

---

## Comparação de Estratégias

| Estratégia         | Velocidade    | Qualidade          | Caso de Uso            |
| ------------------ | ------------- | ------------------ | ---------------------- |
| **Embedding puro** | ⚡⚡⚡ Rápido | ⭐⭐⭐ Bom         | Geral, base grande     |
| **BM25 puro**      | ⚡⚡⚡ Rápido | ⭐⭐ OK            | Palavras-chave claras  |
| **Híbrido**        | ⚡⚡ Médio    | ⭐⭐⭐⭐ Muito bom | Produção (recomendado) |
| **+ Reranking**    | ⚡ Lento      | ⭐⭐⭐⭐⭐ Ótimo   | Qualidade crítica      |

---

## 💻 Exemplo Completo (Python)

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# ===== SETUP =====
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Base de documentos (simulada)
documents = [
    "Medicamentos devem ser armazenados entre 15 e 25°C",
    "A insulina requer temperatura de -20°C a -8°C",
    "Paracetamol é vendido em farmácias",
    "Antibióticos precisam ser protegidos da umidade",
    "Temperatura ambiente é adequada para comprimidos comuns",
]

# ===== PRÉ-PROCESSAMENTO =====
doc_embeddings = [embedding_model.encode(doc) for doc in documents]

# ===== QUERY =====
query = "Como armazenar medicamentos corretamente?"
query_emb = embedding_model.encode(query)

# ===== BUSCA VETORIAL =====
scores = [
    np.dot(query_emb, doc_emb) /
    (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    for doc_emb in doc_embeddings
]

print("=== RANKING INICIAL (Embedding) ===")
ranked_initial = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
for rank, (idx, score) in enumerate(ranked_initial[:3], 1):
    print(f"{rank}. Score: {score:.4f}")
    print(f"   {documents[idx]}\n")

# ===== RERANKING =====
pairs = [[query, documents[idx]] for idx in range(len(documents))]
rerank_scores = reranker.predict(pairs)

print("=== RANKING FINAL (Reranker) ===")
ranked_final = sorted(enumerate(rerank_scores), key=lambda x: x[1], reverse=True)
for rank, (idx, score) in enumerate(ranked_final[:3], 1):
    print(f"{rank}. Score: {score:.4f}")
    print(f"   {documents[idx]}\n")
```

**Output esperado:**

```
=== RANKING INICIAL (Embedding) ===
1. Score: 0.8234
   Medicamentos devem ser armazenados entre 15 e 25°C

2. Score: 0.7891
   A insulina requer temperatura de -20°C a -8°C

3. Score: 0.6234
   Temperatura ambiente é adequada para comprimidos comuns

=== RANKING FINAL (Reranker) ===
1. Score: 0.9123
   Medicamentos devem ser armazenados entre 15 e 25°C

2. Score: 0.8567
   A insulina requer temperatura de -20°C a -8°C

3. Score: 0.5234
   Paracetamol é vendido em farmácias
```

---

## 🎯 Melhores Práticas

### 1. **Normalização de Scores**

```python
def normalize_scores(scores):
    """Escala scores para 0-1"""
    min_score = min(scores)
    max_score = max(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]
```

### 2. **Threshold de Confiança**

```python
def retrieve_relevant(chunks, scores, threshold=0.5):
    """Retorna apenas chunks acima do threshold"""
    return [
        (chunk, score) for chunk, score in zip(chunks, scores)
        if score >= threshold
    ]
```

### 3. **Diversidade**

```python
def diverse_ranking(chunks, scores, max_similar=0.95):
    """Evita chunks muito similares"""
    selected = []
    for chunk, score in sorted(zip(chunks, scores),
                               key=lambda x: x[1], reverse=True):
        # Verifica se já tem algo muito similar
        if not any(cosine_similarity(chunk['embedding'],
                                    sel['embedding']) > max_similar
                   for sel in selected):
            selected.append({'chunk': chunk, 'score': score})
            if len(selected) >= 5:
                break
    return selected
```

### 4. **Contexto (Context Window)**

```python
# Recuperar chunks + vizinhos para manter contexto
retrieved_chunks = [
    chunks[i] for i in top_indices
] + [
    chunks[i+1] for i in top_indices if i+1 < len(chunks)
]
```

---

## 🔧 Banco de Dados Vetorial

As operações de busca acontecem em bases especializadas:

| DB           | Latência | Escalabilidade | Uso             |
| ------------ | -------- | -------------- | --------------- |
| **Weaviate** | ~10ms    | ⭐⭐⭐⭐       | Produção        |
| **Pinecone** | ~50ms    | ⭐⭐⭐⭐⭐     | Cloud (fácil)   |
| **Milvus**   | ~20ms    | ⭐⭐⭐⭐       | Self-hosted     |
| **Qdrant**   | ~5ms     | ⭐⭐⭐⭐       | Rápido          |
| **Chroma**   | ~30ms    | ⭐⭐           | Desenvolvimento |
| **FAISS**    | <1ms     | ⭐⭐⭐         | CPU offline     |

```python
# Exemplo com Weaviate
import weaviate

client = weaviate.Client("http://localhost:8080")

# Search by similarity
response = client.query.get("Document").with_near_vector({
    "vector": query_embedding
}).with_limit(5).do()

# Retorna Top-5 chunks ordenados por score
```

---

## ⚠️ Problemas Comuns

### **1. Problema: "Lost in the Middle"**

LLM ignora informações no meio da lista!

```
Chunks passados para LLM:
[relevante, relevante, MENOS relevante, relevante, MENOS relevante]
                               ↑
                        LLM pode ignorar
```

**Solução**: Passar em ordem reversa ou intercalar

### **2. Problema: Chunks Fragmentados**

```
Query: "Como fazer bolo?"

Chunk 1: "...adicione farinha..."
Chunk 2: "...misture com ovos..."
Chunk 3: "...assar por 30 minutos" ← Contexto perdido!
```

**Solução**: Aumentar tamanho do chunk ou usar "sliding window"

### **3. Problema: Semantic Drift**

```
Query: "medicamentos para dor"
Top-1: "Aspirina é feita com willow bark" (similar, mas não útil)
Top-2: "Paracetamol alivia dor de cabeça" (mais útil)
```

**Solução**: Fine-tunar embeddings com seus dados
