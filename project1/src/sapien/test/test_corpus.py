from sapien.core.corpus import CorpusLoader


def main():
    corpus = CorpusLoader("ptwiki-small.arrow", batch_size=200)

    for i, doc in enumerate(corpus):
        print(f"{i}. {doc.title} -> {len(doc.content)} chars")
        if i == 500:  # número máximo de documentos a mostrar
            break


if __name__ == "__main__":
    main()
