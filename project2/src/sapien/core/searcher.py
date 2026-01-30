import io, json, math ,re
from functools import lru_cache
from pathlib import Path

import orjson
import zstandard as zstd

from neural.rag import RAGGenerator
from neural.reranker import NeuralReranker
from sapien.core.tokenizer import Tokenizer

from neural.query_expander import QueryExpander

# --- Instâncias globais ---
GLOBAL_RERANKER = NeuralReranker()
GLOBAL_RAG = RAGGenerator()


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
    
        # 1) Títulos + metadata
        self._load_titles()
    
        # 2) Snippets pré-gerados
        self.doc_snippets = self._load_snippets()
    
        # 3) Query Expansion (LLM)
        self.expander = QueryExpander()
    
        # 4) Calcular avg_doc_length
        if self.doc_lengths:
            self.avg_doc_length = sum(map(int, self.doc_lengths.values())) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 1.0
    
        # 5) Descomprimir índice (só 1 vez)
        self._prepare_tmp_file()
    
        # 6) Construir lexicon (termo → offset)
        self._build_lexicon()
    
        # 7) Construir mini índice direto (doc → termos)
        self.doc_terms = {}
        self._build_doc_terms()
    
        # 8) SÓ AGORA carregar textos completos
        self.full_doc_offsets = self._load_full_doc_offsets()

    # ----------------- Assignment 2 -----------------

    # Neural Reranking
    def semantic_search(self, query: str, top_k: int = 50):
        """BM25 → Neural Reranking"""

        # 1. BM25 retrieve
        bm25_results = self.search(query, top_k=top_k)

        # Se não houver resultados
        if not bm25_results:
            return []

        # 2. Converter (doc_id, score) -> documentos completos
        docs_full = []
        for doc_id, score in bm25_results:
            doc = {
                "doc_id": doc_id,
                "title": self.doc_titles.get(str(doc_id), "Sem título"),
                "text": self.get_snippet(doc_id),
                "bm25_score": score,
            }
            docs_full.append(doc)

        # 3. Neural reranking
        reranked_docs = GLOBAL_RERANKER.rerank(query, docs_full)

        return reranked_docs

    # RAG
    def answer_query(self, query: str) -> dict:
        """Gera resposta RAG usando o documento mais relevante (BM25 → Reranker → Top-1)."""

        # 1. BM25 → Neural Reranking
        reranked_docs = self.semantic_search(query, top_k=10)

        if not reranked_docs:
            return {"answer": "Não foram encontrados documentos relevantes.", "doc": None}

        # 2. Top-1 documento
        top_doc = reranked_docs[0]
        
        snippet_text, segment_id = self.get_semantic_snippet(query, top_doc["doc_id"])

        answer = GLOBAL_RAG.answer(query, snippet_text)

        return {
            "answer": answer,
            "doc": top_doc,
            "segment_id": segment_id
        }

    # ----------------- Assignment 1 -----------------

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

        print(
            f"Carregados {len(self.doc_titles):,} títulos e {len(self.doc_lengths):,} comprimentos.\n"
        )

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
    
    
    # ------ BONUS POINT - A ---------

    def _load_full_doc_offsets(self):
        path = Path(__file__).resolve().parents[3] / "data" / "full_docs_offsets.json"

        if not path.exists():
            print("[ERROR] full_docs_offsets.json não encontrado.")
            return {}

        with path.open("r", encoding="utf-8") as f:
            offsets = json.load(f)

        print(f"[RAG] Offsets carregados: {len(offsets):,} documentos.")
        return offsets

    def get_full_document(self, doc_id: str):
        offset = self.full_doc_offsets.get(str(doc_id))
        if offset is None:
            return ""
    
        path = Path(__file__).resolve().parents[3] / "data" / "full_docs.jsonl"
        with path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            if not line:
                return ""
    
            obj = json.loads(line)
            return obj.get("text", "")

    def get_semantic_snippet(self, query: str, doc_id: str):
        text = self.get_full_document(doc_id)

        if not text:
            return "Content unavailable.", 0

        segments = re.split(r"(?<=[.!?])\s+", text)
        segments = [s.strip() for s in segments if len(s.strip()) > 30]

        if not segments:
            return text, 0

        tmp_docs = [
            {"doc_id": f"{doc_id}_{i}", "text": seg, "title": "", "segment_id": i+1}
            for i, seg in enumerate(segments)
        ]

        reranked = GLOBAL_RERANKER.rerank(query, tmp_docs)
        best = reranked[0]

        return best["text"], best["segment_id"]
    

    # ------------------------------


    # ------ BONUS POINT - C ---------

    def expanded_search(self, query: str, num_results: int = 10):
        expanded = self.expander.expand(query)

        print(f"[Query Expansion] Original: {query}")
        print(f"[Query Expansion] Expandida: {expanded}")

        # Pesquisa BM25 usando a query expandida
        ranked = self.search(expanded, top_k=num_results)

        results = []
        for doc_id, score in ranked:
            title = self.doc_titles.get(str(doc_id), f"Documento {doc_id}")
            snippet = self.get_snippet(doc_id)

            results.append({
                "id": int(doc_id),
                "title": title,
                "snippet": snippet,
                "score": score,
            })

        return {
            "query_original": query,
            "query_expanded": expanded,
            "results": results
        }

    # ------------------------------


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
        """Cria um pequeno índice direto em memória (doc_id -> termos),
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
