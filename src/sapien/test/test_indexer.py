import shutil
from pathlib import Path

import orjson
import zstandard as zstd

from sapien.core.indexer import build_index
from sapien.core.tokenizer import Tokenizer


def main():
    # Caminhos
    data_path = Path(__file__).resolve().parents[3] / "data" / "ptwiki-small-50000.arrow"
    output_dir = Path(__file__).resolve().parents[3] / "data" / "indexes"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"Ficheiro não encontrado: {data_path}")
        return

    print(f"Corpus: {data_path.name}")

    # Tokenizer configurado
    tokenizer = Tokenizer(
        lowercase=True,
        remove_punctuation=True,
        remove_accents_flag=True,
        min_length=3,
        remove_stopwords=True,
        use_stemming=False,  # <- se quiseres comparar, mete False
        keep_numbers=False,
        keep_alphanum=False,
    )

    # Construção do índice (blocos comprimidos)
    build_index(data_path, output_dir, tokenizer)

    # --- Verificação ---
    term = "portugal"
    merged_postings = {}

    print("\nA verificar todos os blocos...")
    dctx = zstd.ZstdDecompressor()

    # Lê e funde os blocos (modo streaming)
    for block_path in sorted(output_dir.glob("index_block_*.zst")):
        with dctx.stream_reader(block_path.open("rb")) as reader:
            raw = reader.read()
        data = orjson.loads(raw)

        print(f" - Lido {block_path.name} ({len(data['index']):,} termos)")

        for term_key, postings in data["index"]:
            if term_key == term:
                for doc_id, freq in postings:
                    merged_postings[int(doc_id)] = merged_postings.get(int(doc_id), 0) + freq

    # Resultados
    print(f"\nTermo '{term}' encontrado em {len(merged_postings)} documentos no total.")
    if merged_postings:
        print(f"Exemplo de posting list: {dict(list(merged_postings.items())[:5])}")
    else:
        print("Termo não encontrado em nenhum bloco.")


if __name__ == "__main__":
    main()
