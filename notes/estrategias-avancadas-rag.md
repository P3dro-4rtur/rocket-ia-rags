# Estratégias Avançadas de RAG (Retrieval-Augmented Generation)

Além do RAG simples e da estratégia Parent-Child, existem diversas técnicas avançadas para melhorar a precisão, reduzir alucinações e otimizar o uso de tokens. Abaixo estão as principais estratégias utilizadas no mercado:

---

## 1. Re-ranking (Re-rankeamento)
**O Problema:** A busca vetorial (baseada em distância euclidiana ou cosseno) é rápida para grandes volumes, mas pode trazer documentos que são matematicamente similares, mas semanticamente irrelevantes para a resposta.

**A Solução:**
1. Recupera um número maior de documentos (ex: Top 20) usando busca vetorial simples.
2. Passa esses documentos por um modelo de **Cross-Encoder** (como Cohere Rerank ou modelos BGE).
3. O modelo avalia o par (Pergunta, Documento) e atribui uma nota de relevância real.
4. Apenas os Top 3 ou 5 reais são enviados ao LLM.

---

## 2. Multi-Query Retrieval (Expansão de Query)
**O Problema:** Perguntas de usuários costumam ser curtas, ambíguas ou usar termos diferentes dos que estão nos documentos técnicos.

**A Solução:**
1. O LLM recebe a pergunta original e gera **N variações** (sinônimos ou perspectivas diferentes).
2. O sistema faz a busca vetorial para cada uma das variações.
3. Os resultados são combinados (frequentemente usando *Reciprocal Rank Fusion*) para encontrar os documentos que aparecem com mais frequência ou relevância entre todas as buscas.

---

## 3. HyDE (Hypothetical Document Embeddings)
**O Problema:** A busca por similaridade funciona melhor quando comparamos "Resposta com Resposta" do que "Pergunta com Resposta".

**A Solução:**
1. O LLM gera uma **resposta hipotética** para a pergunta do usuário (mesmo que contenha erros).
2. O embedding dessa resposta hipotética é gerado.
3. A busca no banco de dados é feita comparando a resposta hipotética com os documentos reais.
4. Isso ajuda a encontrar documentos que "se parecem" com o que uma resposta correta deveria ser.

---

## 4. Contextual Compression (Compressão de Contexto)
**O Problema:** Chunks grandes contêm informações irrelevantes que gastam "janela de contexto" e podem distrair o modelo.

**A Solução:**
1. Um modelo ou filtro analisa os documentos recuperados.
2. Ele extrai apenas os fragmentos (frases ou parágrafos) que respondem diretamente à pergunta.
3. O LLM final recebe apenas o "suco" da informação, reduzindo custos e aumentando a precisão.

---

## 5. Self-RAG / Self-Reflection
**O Problema:** O modelo tenta responder mesmo quando os documentos não contêm a resposta, gerando alucinações.

**A Solução:**
- O pipeline inclui etapas de auto-crítica:
    - **Is_Relevant:** O modelo avalia se o documento recuperado é realmente útil.
    - **Is_Supported:** O modelo verifica se a resposta gerada é baseada nos fatos do documento.
    - **Is_Useful:** O modelo avalia se a resposta final resolve o problema do usuário.

---

## 6. GraphRAG (RAG Baseado em Grafos)
**O Problema:** A busca vetorial falha em conectar informações que estão em documentos diferentes ou que exigem raciocínio estruturado (ex: "Qual a relação entre o autor X e a empresa Y?").

**A Solução:**
1. Os documentos são processados para extrair entidades e relações (Triplas: Sujeito-Verbo-Objeto).
2. Essas informações formam um **Grafo de Conhecimento**.
3. A busca combina similaridade vetorial com navegação pelos nós do grafo para entender o contexto global da informação.

---

## 7. Agentic RAG (RAG com Agentes)
**O Problema:** Nem toda pergunta precisa da mesma estratégia de busca.

**A Solução:**
- Um **Agente** (LLM com capacidade de decisão) escolhe o fluxo:
    - Se for uma pergunta simples: Busca vetorial direta.
    - Se for uma comparação: Busca em múltiplos documentos e resume.
    - Se não souber: Pesquisa na Web ou consulta um banco SQL.
- O Agente pode "refinar" a busca em um loop até estar satisfeito com a informação encontrada.
