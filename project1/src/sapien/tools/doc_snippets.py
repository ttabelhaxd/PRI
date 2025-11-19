import json
from pathlib import Path
from sapien.core.corpus import CorpusLoader

CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "ptwiki-articles-with-redirects.arrow"
OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data" / "snippets.json"

print(f"A ler o corpus: {CORPUS_PATH}")

corpus = CorpusLoader("ptwiki-articles-with-redirects.arrow")
snippets = {}

for doc in corpus:
    text = doc.content.strip().replace("\n", " ")
    snippets[str(doc.id)] = text[:500] + "..."  # primeiros 500 caracteres

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(snippets, f, ensure_ascii=False, indent=2)

print(f"Snippets gerados: {len(snippets)} documentos → {OUTPUT_PATH}")
