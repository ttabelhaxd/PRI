import io, math, json
from pathlib import Path
import orjson, zstandard as zstd
from functools import lru_cache
from sapien.core.tokenizer import Tokenizer


class Searcher:
    """Motor de pesquisa BM25 com leitura on-demand e cache."""

    def __init__(self, index_path: Path, tokenizer: Tokenizer):
        self.index_path = index_path
        self.tmp_path = index_path.with_suffix(".tmp.jsonl")
        self.tokenizer = tokenizer
        self.lexicon = {}
        self.num_docs = 0
        self.doc_lengths = {}
        self.doc_titles = {}
        self._load_titles()
        self.doc_snippets = self._load_snippets()

        self._load_titles()

        if self.doc_lengths:
            self.avg_doc_length = sum(map(int, self.doc_lengths.values())) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 1.0

        # --- Descomprimir o índice ---
        self._prepare_tmp_file()

        # --- Construir lexicon (termo -> offset) ---
        self._build_lexicon()

        # --- Construir índice direto leve (doc -> termos) ---
        self.doc_terms = {}
        self._build_doc_terms()

    # Descompressão
    def _prepare_tmp_file(self):
        if self.tmp_path.exists():
            print(f"O ficheiro temporário já existe: {self.tmp_path.name}")
            return

        print(f"A descomprimir índice para {self.tmp_path.name} (1x operação)...")
        dctx = zstd.ZstdDecompressor()
        with self.index_path.open("rb") as f_in, self.tmp_path.open("wb") as f_out:
            dctx.copy_stream(f_in, f_out)
        print(f"Descompressão concluída: {self.tmp_path.stat().st_size / 1e9:.2f} GB")

    # Carregar títulos e metadados
    def _load_titles(self):
        """Extrai títulos e comprimentos dos documentos diretamente do índice final."""
        if not self.index_path.exists():
            print(f"Índice não encontrado: {self.index_path}")
            return

        print(f"A carregar cabeçalho de {self.index_path.name} ...")

        dctx = zstd.ZstdDecompressor()
        with self.index_path.open("rb") as f_in:
            with dctx.stream_reader(f_in) as stream:
                text_stream = io.TextIOWrapper(stream, encoding="utf-8")

                try:
                    header = orjson.loads(text_stream.readline().strip())
                except Exception:
                    print("Erro a ler cabeçalho do índice.")
                    return

                self.num_docs = header.get("num_docs", 0)
                self.doc_lengths = header.get("doc_lengths", {}) or {}

                doc_titles = header.get("doc_titles", {})
                if isinstance(doc_titles, list):
                    self.doc_titles = {str(k): v for k, v in doc_titles}
                elif isinstance(doc_titles, dict):
                    self.doc_titles = {str(k): v for k, v in doc_titles.items()}

        print(f"Carregados {len(self.doc_titles):,} títulos e {len(self.doc_lengths):,} comprimentos.\n")

    # Conteudo dos documentos
    def _load_snippets(self):
        """Carrega snippets pré-gerados do ficheiro JSON."""
        snippets_path = Path(__file__).resolve().parents[3] / "data" / "snippets.json"
        if not snippets_path.exists():
            print(f"Snippets não encontrados em {snippets_path}")
            return {}
        with snippets_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Snippets carregados: {len(data):,} documentos.")
            return data

    def get_snippet(self, doc_id: str) -> str:
        """Devolve o snippet associado ao documento (ou uma mensagem padrão)."""
        return self.doc_snippets.get(str(doc_id), "Snippet não disponível.")

    # Lexicon (termo -> offset)
    def _build_lexicon(self):
        print(f"A construir lexicon leve a partir de {self.tmp_path.name} ...")
        with self.tmp_path.open("r", encoding="utf-8") as f:
            header = orjson.loads(f.readline().strip())
            self.num_docs = header.get("num_docs", 0)
    
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    term = next(iter(orjson.loads(line).keys()))
                    self.lexicon[term] = offset
                except Exception:
                    continue
    
    # search similar
    def _build_doc_terms(self, max_terms_per_doc=200):
        """
        Cria um pequeno índice direto em memória (doc_id -> termos),
        usando o lexicon e as postings já existentes.
        """
        print("A construir índice direto leve (doc_id -> termos) ...")
        processed = 0

        for term, offset in list(self.lexicon.items())[:300000]:
            postings = self._load_postings(term)
            if not postings:
                continue

            for doc_id, _ in postings:
                if doc_id not in self.doc_terms:
                    self.doc_terms[doc_id] = set()
                if len(self.doc_terms[doc_id]) < max_terms_per_doc:
                    self.doc_terms[doc_id].add(term)

            processed += 1
            if processed % 50000 == 0:
                print(f"  Processados {processed:,} termos...")

        print(f"Índice direto leve criado para {len(self.doc_terms):,} documentos.")


    # Ler postings on-demand
    @lru_cache(maxsize=2048)
    def _load_postings(self, term: str):
        """Lê do ficheiro apenas a linha do termo pedido."""
        offset = self.lexicon.get(term)
        if offset is None:
            return None

        with self.tmp_path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            if not line:
                return None
            data = orjson.loads(line)
            return data.get(term)

    # Pesquisa BM25
    def search(self, query: str, top_k: int = 10, k1: float = 5.0, b: float = 0.25):
        """Executa pesquisa BM25 em modo on-demand (corrigido para freq)."""
        if self.num_docs == 0:
            return []

        print(f"BM25 params → k1={k1}, b={b}, avg_doc_len={self.avg_doc_length:.2f}")

        query_terms = self.tokenizer.tokenize(query)
        scores = {}

        for term in query_terms:
            postings = self._load_postings(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1e-6)

            for doc_id, data in postings:
                if isinstance(data, int):
                    freq = data
                elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], int):
                    freq = data[0]
                else:
                    freq = len(data)

                doc_len = self.doc_lengths.get(str(doc_id), 1)
                norm = (1 - b) + b * (doc_len / self.avg_doc_length)
                score = idf * ((freq * (k1 + 1)) / (freq + k1 * norm))
                scores[doc_id] = scores.get(doc_id, 0) + score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
