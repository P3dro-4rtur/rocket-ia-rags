# Carregamento de Documentos: PyPDFLoader vs GenericLoader + LanguageParser

> **Referência nos notebooks:** `simple-rag-documents.ipynb`, `parent-rag.ipynb` e `rerank-rag.ipynb` usam `PyPDFLoader`; `code-review-rag.ipynb` usa `GenericLoader` + `LanguageParser`.

---

## O Conceito de Document Loader

Em LangChain, um **Document Loader** lê um arquivo e retorna objetos `Document`:

```python
class Document:
    page_content: str   # O texto extraído
    metadata: dict      # Informações sobre a origem
```

---

## PyPDFLoader: Para Documentos de Texto

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./assets/anexo-projeto-de-lei.pdf', extract_images=False)
pages = loader.load_and_split()
# Retorna 31 Documents — um por página do PDF
```

### Metadados gerados

```python
{
    'source': './assets/anexo-projeto-de-lei.pdf',
    'page': 0,              # Índice 0-based
    'page_label': '1',      # Rótulo 1-based no PDF
    'total_pages': 31,
    'author': 'fredfqd',
    'creator': 'Microsoft Office Word',
    'creationdate': '2023-05-03T18:22:00+00:00',
}
```

### Limitações

O PyPDFLoader trata tudo como texto bruto. Perde formatação de tabelas, não distingue títulos de parágrafos e pode cortar frases em transições de página. Para PDFs de texto contínuo (leis, artigos), funciona bem. Para código-fonte, é inadequado.

---

## GenericLoader + LanguageParser: Para Código-Fonte

```python
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

loader = GenericLoader.from_filesystem(
    os.path.join(repo_path, 'libs/core/langchain_core'),
    glob='**/*',
    suffixes=['.py'],
    exclude=['**/non-utf-8-enconding.py'],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)

documents = loader.load()  # 641 documentos
```

### Como o LanguageParser funciona

Usa **tree-sitter** para parsear a AST (Abstract Syntax Tree) do Python. Cada nó semântico (classe, função) vira um `Document` separado:

```
arquivo.py
├── class MinhaClasse → Document(page_content="class MinhaClasse:...")
│   ├── def __init__ → Document(page_content="def __init__:...")
│   └── def metodo   → Document(page_content="def metodo:...")
└── def func_global  → Document(page_content="def func_global:...")
```

### `parser_threshold=500`

Ignora arquivos com menos de 500 caracteres (arquivos `__init__.py` vazios, stubs mínimos). Reduz ruído sem perder lógica relevante.

---

## Splitters: Genérico vs. Específico de Linguagem

### `RecursiveCharacterTextSplitter` — para texto (todos exceto code-review)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=20,
)
# Hierarquia de separadores: ["\n\n", "\n", " ", ""]
```

### `from_language(Language.PYTHON)` — para código

```python
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2_000,
    chunk_overlap=200
)
# Hierarquia: ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
```

**Por que o splitter de linguagem importa:**

```python
# Sem from_language — corta no meio de uma função:
chunk_1 = "def calcular(valores):\n    total = sum(val"
chunk_2 = "ores)\n    return total / len(valores)"

# Com from_language — respeita limites semânticos:
chunk_1 = "def calcular(valores):\n    total = sum(valores)\n    return total / len(valores)"
```

---

## Clonagem do Repositório Git

```python
from git import Repo

repo = Repo.clone_from(
    'https://github.com/langchain-ai/langchain',
    to_path='./repo_review'
)
# Equivalente a: git clone <url> ./repo_review
```

Metadados gerados pelo `GenericLoader`:
```python
{
    'source': './repo_review/libs/core/langchain_core/runnables/base.py',
    'language': 'python'
}
```

---

## Comparação Final

| Característica       | PyPDFLoader           | GenericLoader + LanguageParser    |
|----------------------|-----------------------|-----------------------------------|
| **Conteúdo ideal**   | PDFs, documentos      | Código-fonte                      |
| **Divisão por**      | Páginas               | Classes, funções, métodos         |
| **Preserva**         | Texto + metadados PDF | Estrutura semântica (AST)         |
| **Splitter ideal**   | `RecursiveCharacter`  | `from_language(Language.PYTHON)`  |
| **Resultado (31pg)** | 31 docs por página    | 641 unidades semânticas           |
| **Após splitting**   | ~31 chunks de 4000c   | 1803 chunks de 2000c completos    |
