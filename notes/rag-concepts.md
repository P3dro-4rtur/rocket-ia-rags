# Conceitos de RAG: Chunk Size e Chunk Overlap

Em sistemas de RAG (Retrieval-Augmented Generation), a divisão do texto em pedaços (chunks) é fundamental para que o modelo de linguagem consiga processar grandes volumes de dados com precisão.

## 1. Chunk Size (Tamanho do Bloco)

É o limite máximo de caracteres ou tokens que cada pedaço de texto terá.

- **Objetivo:** Adaptar o conteúdo ao "context window" (janela de contexto) do modelo de IA.
- **Impacto:**
  - **Pequeno demais:** Perda de contexto semântico (a informação fica fragmentada).
  - **Grande demais:** Pode incluir ruído (informação irrelevante) e consumir muitos tokens desnecessariamente.

## 3. O que é um Token?

Tokens são as unidades básicas que os modelos de linguagem (como o GPT) usam para processar texto.

- **Não é apenas uma palavra:** Um token pode ser uma palavra inteira, uma parte de uma palavra (sub-word), um caractere individual ou até mesmo sinais de pontuação.
- **Estimativa:** Em média, 1.000 tokens equivalem a cerca de 750 palavras.
- **Por que importa?** Os modelos têm limites de "Janela de Contexto" medidos em tokens. Além disso, o custo das APIs (OpenAI, Anthropic) é geralmente baseado na quantidade de tokens processados.

## 2. Chunk Overlap (Sobreposição)

É a quantidade de texto que se repete entre um bloco e o próximo.

- **Objetivo:** Manter a continuidade e o contexto entre as divisões.
- **Impacto:**
  - Garante que uma frase ou definição de código que foi cortada ao final de um bloco apareça completa no início do próximo.
  - Ajuda o sistema de busca (Vector Store) a encontrar blocos que mantenham o sentido completo da informação.

## 4. Granularidade de Memória

A granularidade refere-se ao "nível de detalhe" da sua base de conhecimento recuperável.

- **Alta Granularidade (Chunks Pequenos):** Permite encontrar respostas muito específicas e precisas, mas a IA pode ter dificuldade em entender o "quadro geral" ou a relação entre diferentes partes do código.
- **Baixa Granularidade (Chunks Grandes):** Oferece muito contexto para a IA, facilitando a compreensão de lógicas complexas, mas pode tornar a busca mais lenta, cara e propensa a trazer informações desnecessárias que confundem o modelo.
- **O Equilíbrio:** O objetivo do desenvolvedor de RAG é encontrar o "ponto doce" onde o bloco é pequeno o suficiente para ser específico, mas grande o suficiente para ser autossuficiente (ter sentido próprio).

---

## Resumo para Código (Python/JS)

Ao trabalhar com arquivos de código-fonte:

- O **Chunk Size** deve ser grande o suficiente para conter uma função ou classe média (ex: 1000-2000 caracteres).
- O **Chunk Overlap** deve ser suficiente para não perder o cabeçalho de uma função ou importações importantes (ex: 10% a 15% do tamanho do chunk).

> **Dica:** O uso de splitters específicos de linguagem (`from_language`) ajuda a respeitar a sintaxe (evita cortar no meio de uma palavra ou indentação vital).
