from pathlib import Path

import pyarrow.dataset as ds


def main():
    # data_path = Path(__file__).resolve().parents[2] / "data" / "ptwiki-articles-with-redirects.arrow"
    data_path = Path(__file__).resolve().parents[2] / "data" / "ptwiki-small.arrow"

    dataset = ds.dataset(str(data_path), format="arrow")
    table = dataset.to_table().slice(0, 500)

    print("Columns:", table.column_names, "\n")
    for row in table.to_pylist():
        print("Title:", row["title"])
        print("Text:", row["text"])
        # print("Text:", row["text"][:200])   # Primeiros 200 caracteres do texto
        print("Redirect:", row["redirect"])
        print("Out links:", row["out_links"][:3])
        print("-" * 50)


if __name__ == "__main__":
    main()
