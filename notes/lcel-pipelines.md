# LangChain Expression Language (LCEL): O Operador Pipe e Runnables

> **Referência nos notebooks:** `parent-rag.ipynb` e `rerank-rag.ipynb` usam LCEL extensivamente com `RunnableParallel`, `RunnablePassthrough` e o operador `|`.

---

## O Problema Que o LCEL Resolve

Antes do LCEL, montar um pipeline RAG exigia código imperativo e repetitivo:

```python
# Abordagem antiga (sem LCEL)
def pipeline(question):
    # 1. Recuperar contexto
    docs = retriever.get_relevant_documents(question)
    
    # 2. Formatar prompt manualmente
    context_text = "\n".join([d.page_content for d in docs])
    prompt_text = f"Contexto: {context_text}\n\nPergunta: {question}"
    
    # 3. Chamar LLM
    ai_message = llm.predict(prompt_text)
    
    # 4. Extrair string
    return ai_message.content
```

Com LCEL, o mesmo pipeline se torna:

```python
# Abordagem moderna (com LCEL)
chain = setup_retrieval | prompt | llm | StrOutputParser()
answer = chain.invoke(question)
```

---

## O Operador `|` (Pipe)

O operador `|` encadeia componentes em sequência. A saída de um componente se torna a entrada do próximo.

```
entrada → [componente_1] → resultado_1 → [componente_2] → resultado_2 → ...
```

### Regra fundamental: compatibilidade de tipos

Cada componente deve aceitar o tipo de saída do componente anterior:

| Componente          | Recebe                   | Retorna               |
|---------------------|--------------------------|-----------------------|
| `RunnableParallel`  | `str` (a query)          | `dict` com chaves     |
| `ChatPromptTemplate`| `dict` com placeholders  | `ChatPromptValue`     |
| `ChatOpenAI`        | `ChatPromptValue`        | `AIMessage`           |
| `StrOutputParser`   | `AIMessage`              | `str`                 |

---

## RunnableParallel: Executando Múltiplos Caminhos

### O que é

`RunnableParallel` recebe **uma única entrada** e a distribui para múltiplos runnables simultaneamente, coletando os resultados em um dicionário.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

setup_retrieval = RunnableParallel({
    'query': RunnablePassthrough(),        # Passa a entrada sem alteração
    'context': parent_document_retriever,  # Busca documentos para a entrada
})
```

### Fluxo visual

```
                          ┌─ RunnablePassthrough() ──────── 'query': "qual é a IA?"
"qual é a IA?" ──────────┤
                          └─ parent_document_retriever ──── 'context': [Doc1, Doc2, ...]

Resultado: {'query': 'qual é a IA?', 'context': [Doc1, Doc2, ...]}
```

### Por que isso é necessário?

O `ChatPromptTemplate` precisa de **ambos** ao mesmo tempo:
- O texto da query (para o placeholder `{query}`)
- Os documentos de contexto (para o placeholder `{context}`)

Sem o `RunnableParallel`, precisaríamos de código manual para reunir essas informações.

---

## RunnablePassthrough: O "Fio" da Chain

`RunnablePassthrough()` é o componente mais simples do LCEL: ele recebe qualquer entrada e a retorna **idêntica**, sem modificações.

```python
# Exemplo didático
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()
result = passthrough.invoke("qualquer coisa")
# result == "qualquer coisa"
```

**Por que precisamos disso?**

No `RunnableParallel`, a entrada vai para **todos** os runnables. O retriever transforma a query em documentos. O `RunnablePassthrough` garante que a query original também chegue ao prompt, sem ser transformada.

---

## ChatPromptTemplate: Dois Modos de Criação

Os notebooks usam dois modos diferentes de criação de prompts:

### `from_template` (usado em `parent-rag.ipynb` e `rerank-rag.ipynb`)

```python
TEMPLATE = """
Você é um especialista. Responda usando o contexto.

Query:
{query}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)
```

- Cria um prompt de **role único** (sem separação system/user)
- Mais simples, mas menos controle sobre o comportamento do modelo
- Placeholders são definidos com `{nome_da_variavel}`

### `from_messages` (usado em `code-review-rag.ipynb`)

```python
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um revisor de código. Contexto:\n{context}'),
    ('user', '{input}')
])
```

- Cria um prompt com **múltiplos roles** (system + user)
- O role `system` define o comportamento/persona do modelo
- O role `user` representa a pergunta do usuário
- Oferece mais controle sobre como o LLM processa a informação

---

## StrOutputParser: Desembrulhando a Resposta

O LLM retorna um objeto `AIMessage`, não uma string simples:

```python
# Sem o parser:
result = llm.invoke(prompt)
# result = AIMessage(content='Olá!', response_metadata={...}, ...)
# Para acessar o texto: result.content

# Com o StrOutputParser:
chain = llm | StrOutputParser()
result = chain.invoke(prompt)
# result = 'Olá!'  ← Já é uma string limpa
```

---

## A Chain Completa do Parent RAG

```python
# Montagem do pipeline completo
parent_chain_retrival = setup_retrival | rag_prompt | llm | output_parser
```

### Fluxo de dados passo a passo

```
ENTRADA: "Quais os principais riscos do marco legal de IA?"
    │
    ▼
[RunnableParallel]
    ├── 'query': "Quais os principais riscos..."
    └── 'context': [chunk_pai_1 (4000 chars), chunk_pai_2 (4000 chars), ...]
    │
    ▼ {'query': '...', 'context': [...]}
    │
[ChatPromptTemplate.from_template]
    → "Você é especialista... Query: Quais os riscos... Context: CAPÍTULO I..."
    │
    ▼ ChatPromptValue (prompt formatado)
    │
[ChatOpenAI - gpt-3.5-turbo]
    → Processa o prompt, acessa os documentos como contexto
    │
    ▼ AIMessage(content='Os principais riscos são...')
    │
[StrOutputParser]
    → Extrai apenas o .content
    │
    ▼
SAÍDA: "Os principais riscos são..."
```

---

## LCEL vs. Abordagem Antiga

| Característica       | Código Imperativo         | LCEL com `|`                  |
|----------------------|---------------------------|-------------------------------|
| **Legibilidade**     | Difícil de seguir         | Fluxo visual claro            |
| **Streaming**        | Implementação manual      | Nativo (`.stream()`)          |
| **Paralelismo**      | Async manual              | `RunnableParallel` automático |
| **Composição**       | Acoplamento alto          | Componentes intercambiáveis   |
| **Rastreamento**     | Logs manuais              | LangSmith integrado           |
| **Reutilização**     | Baixa (lógica misturada)  | Alta (chain é um objeto)      |

---

## Modos de Invocação de uma Chain

Qualquer chain LCEL suporta múltiplos modos de execução:

```python
chain = setup_retrival | rag_prompt | llm | output_parser

# 1. Síncrono (bloqueia até terminar)
result = chain.invoke("pergunta")

# 2. Streaming (imprime tokens conforme chegam)
for chunk in chain.stream("pergunta"):
    print(chunk, end="", flush=True)

# 3. Assíncrono
result = await chain.ainvoke("pergunta")

# 4. Batch (múltiplas perguntas de uma vez)
results = chain.batch(["pergunta 1", "pergunta 2", "pergunta 3"])
```

> **Dica:** Todas as chains LCEL herdam esses métodos automaticamente — você não precisa implementar nenhum deles.
