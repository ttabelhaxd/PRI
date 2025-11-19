import io
from itertools import islice
from pathlib import Path

import orjson
import zstandard as zstd

path = Path("../data/indexes_big_data/merged_batched/merged_index_final.zst")

dctx = zstd.ZstdDecompressor()
with dctx.stream_reader(path.open("rb")) as reader:
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    header_line = text_stream.readline()
    header = orjson.loads(header_line.strip())
    print(f"Header: {header}\n")

    text_stream.readline()

    print("Primeiros termos do índice:\n")
    for line in islice(text_stream, 5):
        line = line.strip()
        if not line:
            continue
        term_entry = orjson.loads(line)
        for term, postings in term_entry.items():
            print(f"→ {term:<20} ({len(postings)} docs)")
            print(f"   {postings[:3]}")
