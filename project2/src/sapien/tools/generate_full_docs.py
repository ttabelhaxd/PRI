from pathlib import Path
import json
from sapien.core.corpus import CorpusLoader

OUTPUT = Path(__file__).resolve().parents[3] / "data" / "full_docs.jsonl"

loader = CorpusLoader(
    file_name="ptwiki-articles-with-redirects.arrow",
    batch_size=10000
)

with OUTPUT.open("w", encoding="utf-8") as f:
    for doc in loader:
        obj = {
            "id": doc.id,
            "text": doc.content
        }
        f.write(json.dumps(obj) + "\n")

print("full_docs.jsonl criado com sucesso!")
