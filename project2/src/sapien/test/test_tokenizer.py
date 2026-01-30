from sapien.core.tokenizer import Tokenizer


def main():
    text = """
        Olá! Este é um exemplo de texto para testar o Tokenizer. Inclui URLs
        como https://example.com,também inclui acrónimos como U.S.A. e ainda vários
        exemplos de pontuação!!!?+&/()
    """

    print("\nTexto Original:\n", text)

    print("\nTokens (sem remover acentos):")
    t1 = Tokenizer()
    print(t1.tokenize(text))

    print("\nTokens (com remoção de acentos):")
    t2 = Tokenizer(remove_accents_flag=True)
    print(t2.tokenize(text))

    print("\nTokens (sem remoção de acentos e stemmer):")
    t3 = Tokenizer(use_stemming=True)
    print(t3.tokenize(text))

    print("\nTokens (com remoção de acentos e stemmer):")
    t4 = Tokenizer(remove_accents_flag=True, use_stemming=True)
    print(t4.tokenize(text))


if __name__ == "__main__":
    main()
