import argparse
import json
from pathlib import Path

from sapien.core.corpus import CorpusLoader


def main():
    parser = argparse.ArgumentParser(description="Gerar snippets de documentos do corpus.")

    default_corpus_path = (
        Path(__file__).resolve().parents[3] / "data" / "ptwiki-articles-with-redirects.arrow"
    )
    default_output_path = Path(__file__).resolve().parents[3] / "data" / "snippets.json"

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(default_corpus_path),
        help=f"Caminho para o arquivo de corpus (padrão: {default_corpus_path})",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(default_output_path),
        help=f"Caminho para o arquivo de saída (padrão: {default_output_path})",
    )

    args = parser.parse_args()

    corpus_path = Path(args.input)
    output_path = Path(args.output)

    print(f"A ler o corpus: {corpus_path}")

    corpus = CorpusLoader(str(corpus_path))
    snippets = {}

    for doc in corpus:
        text = doc.content.strip().replace("\n", " ")
        snippets[str(doc.id)] = text[:500] + "..."  # primeiros 500 caracteres

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snippets, f, ensure_ascii=False, indent=2)

    print(f"Snippets gerados: {len(snippets)} documentos → {output_path}")


if __name__ == "__main__":
    main()
