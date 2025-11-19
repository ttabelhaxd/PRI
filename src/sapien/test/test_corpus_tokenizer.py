from sapien.core.corpus import CorpusLoader
from sapien.core.tokenizer import Tokenizer


def main():
    corpus = CorpusLoader("ptwiki-small.arrow", batch_size=100)

    tokenizer = Tokenizer(
        remove_accents_flag=False,
        remove_stopwords=True,
        use_stemming=True,
        min_length=2,
    )

    for i, doc in enumerate(corpus):
        tokens = tokenizer.tokenize(doc.content)
        print("--------------------------------------------------")
        print(f"Documento {i} — {doc.title}")
        print(f"Total de tokens: {len(tokens)}")
        print("Primeiros 30 tokens:")
        print(" ".join(tokens[:30]))
        print("--------------------------------------------------\n")

        if i >= 4:
            break


if __name__ == "__main__":
    main()
