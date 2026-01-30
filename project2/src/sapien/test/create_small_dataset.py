from pathlib import Path

import pyarrow.dataset as ds
from pyarrow import ipc


def main():
    base_dir = Path(__file__).resolve().parents[3] / "data"
    original_path = base_dir / "ptwiki-articles-with-redirects.arrow"
    small_path = base_dir / "ptwiki-small-50000.arrow"

    if not original_path.exists():
        print(f"Ficheiro não encontrado: {original_path}")
        return

    dataset = ds.dataset(str(original_path), format="arrow")

    table = dataset.to_table().slice(0, 50000)
    print(f"Lidos {table.num_rows} documentos")

    with ipc.new_file(small_path.open("wb"), table.schema) as writer:
        writer.write_table(table)

    print(f"Dataset reduzido guardado em: {small_path.resolve()}")


if __name__ == "__main__":
    main()
