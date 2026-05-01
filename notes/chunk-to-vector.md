# De Chunk para Vetor em RAG

## O Processo Geral

Um **chunk** (trecho de texto) é convertido em um **vetor numérico** através de um **modelo de embedding**. Aqui está o fluxo:

```
Texto: "A temperatura hoje é 25°C"
        ↓
[Tokenização]
        ↓
["A", "temperatura", "hoje", "é", "25", "°C"]
        ↓
[Embedding Model]
        ↓
[0.234, -0.891, 0.456, ..., 0.123]  ← Vetor (1536 ou 4096 dimensões)
```

## Como Funciona Internamente

### 1. **Tokenização**

O texto é quebrado em **tokens** (palavras, subpalavras ou caracteres):

```python
# Exemplo com modelo BERT
texto = "25 graus"
tokens = ["25", "gr", "##aus"]  # Subword tokenization
token_ids = [100, 2345, 6789]   # Mapeado para IDs numéricos
```

### 2. **Embedding Layer**

Cada token é convertido em um vetor através de uma **tabela de embeddings** (matriz treinada):

```python
# Tabela de embeddings (vocab_size × embedding_dim)
embedding_table = [
    [0.1, -0.5, 0.3, ...],   # token 0
    [0.2, 0.1, -0.4, ...],   # token 1
    [0.9, 0.2, 0.1, ...],    # token 100 ("25")
    ...
]

# Lookup
vetor_do_token = embedding_table[100]  # → [0.9, 0.2, 0.1, ...]
```

### 3. **Processamento de Contexto**

Os vetores dos tokens são processados por **Transformers** (self-attention) para capturar relacionamentos:

```
Token "25"  →  [0.9, 0.2, 0.1, ...]
                    ↓ (Self-Attention)
               [0.7, 0.3, 0.2, ...] (contextualizando com "graus")
```

### 4. **Agregação Final**

O vetor do **chunk inteiro** é geralmente a:

- **Média** de todos os tokens: `(token1 + token2 + ... + tokenN) / N`
- **Último token CLS** (em BERT)
- **Pooling** sofisticado

```python
chunk_vetor = sum(token_vetores) / len(token_vetores)
# Resultado: vetor de ~768 ou 1536 dimensões
```

---

## 🔢 Como Números são Tratados

Números **não são especiais**. Veja:

### Opção 1: Como Tokens de Texto

```
"25°C"
  ↓
tokens: ["25", "°", "C"]
  ↓
cada um vira um vetor através da embedding table
```

**Problema**: O número "25" é só um token entre milhares. O modelo não "entende" que é um número.

### Opção 2: Normalização Numérica (Melhor Prática)

Alguns sistemas **extraem e normalizam** números:

```python
texto = "A temperatura é 25°C e a pressão é 1013 hPa"

# Extração numérica
numeros = [25, 1013]

# Normalização (ex: log-scaling, z-score)
normalizados = [log(25), log(1013)]  # → [3.22, 6.92]

# Incluir no vetor final como features adicionais
chunk_vetor = [embeddings_texto, 3.22, 6.92]
```

### Opção 3: Tokens Especiais para Números

Alguns tokenizadores criam tokens especiais:

```python
# ELECTRA, RoBERTa com número awareness
tokens = ["<NUM:25>", "°C"]  # Token especial para número

# Durante pré-treinamento, o modelo aprendeu:
# - "<NUM:25>" está próximo de "temperatura"
# - "<NUM:1013>" está próximo de "pressão"
```

---

## Modelos Comuns de Embedding

| Modelo                               | Dimensão      | Treina Números? |
| ------------------------------------ | ------------- | --------------- |
| **OpenAI text-embedding-3-small**    | 512           | Parcialmente    |
| **text-embedding-3-large**           | 3072          | Parcialmente    |
| **Sentence-BERT (all-MiniLM-L6-v2)** | 384           | Básico          |
| **BGE-large-zh**                     | 1024          | Melhor          |
| **ColBERT**                          | 128 por token | Especializado   |

---

## Exemplo Prático (Pseudocódigo)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# Dimensão: 384

chunk = "O paciente tem 42 anos, pressão 120/80, glicemia 95 mg/dL"

# Internamente:
# 1. Tokeniza: ['O', 'pac', '##iente', '42', 'anos', ...]
# 2. Embedding lookup: cada token → vetor 384-dim
# 3. Self-attention: captura que "42" relaciona com "anos"
# 4. Pooling (média):
#    vetor_final = mean([v_O, v_pac, ..., v_42, ..., v_glicemia, ...])
#    resultado: vetor 384-dimensional

embedding = model.encode(chunk)
print(embedding.shape)  # (384,)
print(embedding)        # [0.234, -0.891, 0.456, ...]
```

---

## O Problema Real com Números

❌ **Embeddings genéricos falham com**:

- Números específicos fora do treino ("temperatura 1500°C")
- Valores que precisam comparação numérica ("25 > 20"?)
- Datas e timestamps
- IDs e códigos

✅ **Soluções**:

1. **Separar pipeline**: texto → embedding | números → features normalizadas
2. **Usar RAG híbrido**: BM25 (keyword) + vetorial
3. **Fine-tuning**: treinar o modelo com seus dados numéricos específicos
4. **Embedding estruturado**: converter "temperatura: 25" em `{campo: "temperatura", valor: 25, embedding_campo, embedding_contexto}`
