# Diferença entre Simple RAG e Code Review RAG

A diferença crucial entre estas duas abordagens não está apenas no "assunto" (texto vs. código), mas na **sofisticação da arquitetura** e na **inteligência do contexto**.

O **Simple RAG** funciona como uma busca por similaridade básica, enquanto o **Code Review RAG** atua como um analista estrutural.

---

## 1. Entendimento Estrutural (Parser vs. Texto)

*   **Simple RAG:** Utiliza carregadores genéricos como `PyPDFLoader`. Ele trata o conteúdo como uma "massa de texto". A divisão em blocos (chunking) é feita puramente por contagem de caracteres ou quebras de linha simples.
*   **Code Review RAG:** Utiliza o `GenericLoader` com `LanguageParser` e `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`. 
    *   **Por que importa?** Ele entende a sintaxe do Python. Ele reconhece o que é uma função, uma classe ou um bloco de importação, garantindo que o LLM receba blocos de código com integridade lógica, sem "cortar" um loop ou uma definição no meio.

## 2. Diversidade na Busca (MMR vs. Similaridade Simples)

*   **Simple RAG:** Geralmente busca os "top-K" documentos mais parecidos semanticamente. Se os resultados forem muito redundantes, o LLM recebe informação repetida.
*   **Code Review RAG:** Frequentemente utiliza **MMR (Maximal Marginal Relevance)**.
    *   **Por que importa?** O MMR busca documentos que sejam relevantes à pergunta, mas também **diversos entre si**. Para uma revisão de código, é vital que a IA veja diferentes partes do sistema (ex: a definição da classe, o método principal e os utilitários) em vez de 8 versões quase idênticas do mesmo padrão.

## 3. O Papel das Chains (LCEL)

A diferença na forma como as Chains são utilizadas reflete a evolução do framework LangChain:

*   **Orquestração Profissional:** No Simple RAG, muitas vezes fazemos o trabalho manual (chamar o retriever, filtrar resultados e passar para o LLM). No Code Review, utilizamos a `retrieval_chain`, que automatiza todo o pipeline.
*   **Encapsulamento:** A `retrieval_chain` encapsula a lógica de:
    1. Receber a pergunta.
    2. Consultar o retriever.
    3. Formatar o prompt com o contexto.
    4. Chamar o LLM.
    5. Retornar a resposta estruturada.
*   **Modularidade:** O uso de Chains (especialmente via LCEL) permite adicionar facilmente novos passos, como filtros de segurança, memória de conversa ou ferramentas externas (tools), sem quebrar a lógica principal.

---

### Resumo Comparativo

| Característica | Simple RAG | Code Review RAG |
| :--- | :--- | :--- |
| **Loader** | PyPDFLoader (Texto Puro) | GenericLoader + LanguageParser (Estrutural) |
| **Splitter** | Caracteres Genéricos | Sintaxe de Linguagem (Python/JS/etc) |
| **Busca** | Similaridade Simples | MMR (Relevância + Diversidade) |
| **Abordagem** | Funções Manuais | Chains Orquestradas (`retrieval_chain`) |
| **Foco** | Recuperação de Informação | Análise Lógica e Contextual |
