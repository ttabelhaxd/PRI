import re
import unicodedata

import nltk
import snowballstemmer
from nltk.corpus import stopwords

# garantir stopwords PT
try:
    _ = stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords")

# regex pré-compiladas (performance)
URL_EMAIL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
PUNCT_RE = re.compile(r"[^\w\s]")
ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
TOKEN_RE = re.compile(r"\b\w+\b")


def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_acronyms(text: str) -> str:
    return ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), text)


def clean_urls_emails(text: str) -> str:
    return URL_EMAIL_RE.sub(" ", text)


class Tokenizer:
    """Tokenizador configurável e eficiente para PT."""

    _cached_stopwords = None

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_accents_flag: bool = False,
        min_length: int = 2,
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        keep_numbers: bool = True,
        keep_alphanum: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_accents_flag = remove_accents_flag
        self.min_length = min_length
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.keep_numbers = keep_numbers
        self.keep_alphanum = keep_alphanum

        # cache de stopwords global
        if remove_stopwords:
            if Tokenizer._cached_stopwords is None:
                Tokenizer._cached_stopwords = set(stopwords.words("portuguese"))
            self.stopwords = Tokenizer._cached_stopwords
        else:
            self.stopwords = set()

        if use_stemming:
            if not hasattr(Tokenizer, "_stem_cache"):
                Tokenizer._stem_cache = {}
            if not hasattr(Tokenizer, "_stemmer"):
                Tokenizer._stemmer = snowballstemmer.stemmer("portuguese")

            self.stemmer = Tokenizer._stemmer
        else:
            self.stemmer = None


        # apontadores para regex (micro-otimização)
        self.re_punct = PUNCT_RE
        self.re_acronym = ACRONYM_RE
        self.re_url_email = URL_EMAIL_RE
        self.re_token = TOKEN_RE

    def tokenize(self, text: str) -> list[str]:
        """Processa o texto e devolve tokens normalizados."""
        text = self.re_url_email.sub(" ", text)
        text = self.re_acronym.sub(lambda m: m.group(0).replace(".", ""), text)

        if self.lowercase:
            text = text.lower()
        if self.remove_accents_flag:
            text = remove_accents(text)
        if self.remove_punctuation:
            text = self.re_punct.sub(" ", text)

        tokens = self.re_token.findall(text)

        filtered = [
            t
            for t in tokens
            if len(t) >= self.min_length
            and (self.keep_numbers or not t.isdigit())
            and (self.keep_alphanum or t.isalpha())
            and (not self.remove_stopwords or t not in self.stopwords)
        ]

        if self.stemmer is not None:
            stem_cache = Tokenizer._stem_cache
            stem = self.stemmer.stemWord
            result = []
            for t in filtered:
                if t in stem_cache:
                    result.append(stem_cache[t])
                else:
                    s = stem(t)
                    stem_cache[t] = s
                    result.append(s)
            filtered = result

        return filtered
