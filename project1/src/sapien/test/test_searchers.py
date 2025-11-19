from pathlib import Path
from sapien.core.tokenizer import Tokenizer
from sapien.core.searcher import Searcher

# --- Configuração do Tokenizer ---
tokenizer = Tokenizer(
    lowercase=True,
    remove_punctuation=True,
    remove_accents_flag=True,
    min_length=3,
    remove_stopwords=True,
    use_stemming=True,
    keep_numbers=False,
    keep_alphanum=False,
)

# --- Caminho para o índice ---
index_path = Path("../data/indexes/merged_index.zst")

# --- Criar o Searcher ---
searcher = Searcher(index_path, tokenizer)

# --- Query de teste ---
query = "Porgal"
results = searcher.search(query, top_k=10)

print(f"\nQuery: '{query}'")
for rank, (doc_id, score) in enumerate(results, 1):
    title = searcher.doc_titles.get(str(doc_id), f"Documento {doc_id}")
    print(f"{rank:2d}. {title} — score={score:.4f}")
