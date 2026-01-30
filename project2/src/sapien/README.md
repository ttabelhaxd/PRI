# Notas da primeira apresentação — Projeto PRI

## Módulos Desenvolvidos

---

### **CorpusLoader**
**Função:** leitura e limpeza eficiente do dataset.

- Filtros para remover redericionamentos e documentos vazios;
- Leitura em blocos (**batch_size de 10000** -> *leitura de 10000 documentos em cada batch*);
- Iterador com *yield* para economizar memoria -> processar os documentos de forma incremental e eficiente;

---

### **Tokenizer**
**Função:** normalização e tokenização do texto.

- Limpeza de **URLs, emails e acrónimos**
- **Remoção de acentos e pontuação**
- **Lowercasing**
- **Stopwords (via nltk)**
- **Stemming** (`snowballstemmer`, mais rápido e leve)

**Micro-otimizações:**
- Regex **pré-compiladas** para aumentar a eficiência
- **Caches globais** de stopwords e stemmer → sem overhead de inicialização

---

### **Indexer**
**Função:** construção do **Inverted Index**.

- A função principal `build_index()` processa o corpus nos **blocos configurados de documentos** e grava ficheiros comprimidos (`.zst`)
- **Indexação em blocos**:
  - Cada bloco é gravado como um ficheiro independente (`index_block_X.zst`)
  - Segmentação que melhora **escalabilidade** e evita **sobrecarga de RAM**
- **Compressão com Zstandard** (`level=1`, `threads=4`)
- **Serialização com orjson** → leve e **muito mais rápida** que o `json` nativo
- **`batch_size ≈ 18 800`** → corresponde ao número de documentos processados e indexados antes de criar um novo bloco
  - calculado dinamicamente consoante a **RAM disponível** e é ajustado a cada PC
- **Paralelização da tokenização** (`multiprocessing.Pool`)

---

### **Interação entre os módulos**

| Componente | Função | Batch |
|-------------|--------|--------|
| **CorpusLoader** | Lê o dataset e gera documentos | 10 000 docs/bloco |
| **Indexer** | Processa e grava os índices | ~18 800 docs/bloco |

**Nota final:** o `CorpusLoader` lê os dados de forma eficiente, e o `Indexer` cria blocos adaptativos — maximizando a eficiencia e a velocidade e minimizando o uso de memória.

---

## Testes Desenvolvidos

### **`test_corpus.py`**
- Verifica a leitura e limpeza do corpus.

### **`test_corpus_tokenizer.py`**
- Valida o processo de tokenização, remoção de acentos, stopwords e stemming.

### **`test_indexer.py`**
- Executa a construção completa do índice.
- Gere e grava blocos comprimidos (`.zst`).

---

## Últimas Melhorias (`branch best_results_pcMAX`)

- **Batchs adaptativos:** o tamanho dos blocos de indexação ajusta-se automaticamente à **RAM disponível**.
- **Paralelização otimizada:** aproveitamento até **90% dos núcleos da CPU**.
- **Compressão multi-thread (`zstd`)** para escrita em disco.
- **Serialização binária (`orjson`, implementado em Rust):**
  - Mais leve e rápida que o `json` nativo;
  - Utilizada para converter e gravar o índice antes da compressão:

No projeto:
```python
orjson.dumps(data)   # converte o índice (dict) em bytes JSON ultra-rápidos
zstd.compress(...)   # comprime os bytes para reduzir o espaço em disco
```

---

## Próximos Passos

- Implementação do **Searcher**
- **CLI**
- **logger**

---

## Resumo

| Módulo | Função | Destaques |
|--------|---------|------------|
| **CorpusLoader** | Leitura incremental do corpus | Batchs (10 000 docs), filtragem  |
| **Tokenizer** | Normalização do texto | Regex otimizadas, caching e stemming |
| **Indexer** | Construção do índice | Paralelização + compressão binária (`zstd` + `orjson`) |
| **Testes** | Validação | - |

---
