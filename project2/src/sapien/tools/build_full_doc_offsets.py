import json
from pathlib import Path

base = Path(__file__).resolve().parents[3]
docs_path = base / "data" / "full_docs.jsonl"
out_path = base / "data" / "full_docs_offsets.json"

offsets = {}

with docs_path.open("r", encoding="utf-8") as f:
    while True:
        offset = f.tell()
        line = f.readline()
        if not line:
            break

        obj = json.loads(line)
        offsets[str(obj["id"])] = offset

with out_path.open("w", encoding="utf-8") as f:
    json.dump(offsets, f)

print(f"Offsets gerados para {len(offsets)} documentos.")
