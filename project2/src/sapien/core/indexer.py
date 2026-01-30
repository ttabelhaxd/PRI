import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import orjson
import psutil
import zstandard as zstd

from sapien.core.corpus import CorpusLoader
from sapien.core.tokenizer import Tokenizer


class InvertedIndex:
    """Estrutura principal de índice invertido (pode representar um bloco)."""

    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_lengths = {}
        self.doc_titles = {}
        self.num_docs = 0

    def add_document(self, doc_id: str, tokens: list[str]):
        term_freqs = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        self.num_docs += 1
        for term, freq in term_freqs.items():
            self.index[term][doc_id] += freq

    def save(self, path: Path):
        """Guardar índice em formato binário comprimido."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "index": self.index,
            "doc_lengths": self.doc_lengths,
            "num_docs": self.num_docs,
        }
        cctx = zstd.ZstdCompressor(level=1)
        compressed = cctx.compress(orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS))
        path.write_bytes(compressed)
        print(f"Guardado: {path.name} ({len(self.index):,} termos)")

    @classmethod
    def load(cls, path: Path):
        # leitura descomprimida
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(path.read_bytes())
        data = orjson.loads(raw)
        idx = cls()
        idx.index = defaultdict(dict, data["index"])
        idx.doc_lengths = data["doc_lengths"]
        idx.num_docs = data["num_docs"]
        return idx


def build_index(
    data_path: Path, output_dir: Path, tokenizer: Tokenizer, batch_size: int | None = None
):
    """Cria vários ficheiros de índice parcial (sem merge final)."""

    num_cpus = os.cpu_count() or 4
    try:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        total_ram_gb = 8  # valor seguro

    # ~90% dos CPUs para tokenização
    num_procs = max(1, int(num_cpus * 0.9))

    if batch_size is None:
        batch_size = int((total_ram_gb / 8) * 10000)
        batch_size = min(max(batch_size, 5000), 20000)

    num_compress_threads = max(2, num_cpus // 4)

    print("\nConfiguração automática:")
    print(f"   - CPUs disponíveis: {num_cpus}")
    print(f"   - CPUs usados: {num_procs}")
    print(f"   - RAM total: {total_ram_gb:.1f} GB")
    print(f"   - batch_size: {batch_size}")
    print(f"   - threads de compressão: {num_compress_threads}\n")

    corpus = CorpusLoader(str(data_path), batch_size=batch_size)
    partial_index = defaultdict(list)
    index = InvertedIndex()
    block_num = 0
    start_time = time.time()

    # Compressão paralela
    compressor = zstd.ZstdCompressor(level=1, threads=num_compress_threads)

    # VAMOS TESTANDO PARA PERCEBER QUAL O NUMERO QUE PODEMOS TER - CPU DÁ GAZ
    # criação da pool de processos
    # pool = Pool(processes=14)
    pool = Pool(processes=num_procs)

    # Alternativa - usar todos os CPUs disponíveis e evitar sobrecarga
    # num_cpus = os.cpu_count() or 2
    # pool = Pool(processes=max(1, num_cpus // 2))

    docs_batch = []

    def process_batch(docs):
        """Tokeniza e indexa um batch completo."""
        texts = [d.content for d in docs]
        doc_ids = [str(d.id) for d in docs]

        # tokenização paralela
        tokenized_docs = pool.map(tokenizer.tokenize, texts)

        for doc, tokens in zip(docs, tokenized_docs):
            doc_id = str(doc.id)
            term_freqs = Counter(tokens)
            index.doc_lengths[doc_id] = len(tokens)
            index.doc_titles[doc_id] = doc.title
            index.num_docs += 1

            for term, freq in term_freqs.items():
                partial_index[term].append((doc_id, freq))

    for doc in corpus:
        docs_batch.append(doc)
        if len(docs_batch) >= batch_size:
            process_batch(docs_batch)
            docs_batch.clear()

            # guardamos o bloco sempre que acumulamos batch_size documentos
            block_path = output_dir / f"index_block_{block_num}.zst"
            block_data = {
                "index": list(partial_index.items()),
                "doc_lengths": list(index.doc_lengths.items()),
                "doc_titles": list(index.doc_titles.items()),
                "num_docs": index.num_docs,
            }
            with compressor.stream_writer(block_path.open("wb")) as f:
                f.write(orjson.dumps(block_data))

            print(f"Bloco {block_num} guardado ({len(partial_index):,} termos)")
            partial_index.clear()
            index.doc_lengths.clear()
            index.doc_titles.clear()
            block_num += 1

    # caso exista, para processar último batch
    if docs_batch:
        process_batch(docs_batch)

    if partial_index:
        block_path = output_dir / f"index_block_{block_num}.zst"
        block_data = {
            "index": list(partial_index.items()),
            "doc_lengths": list(index.doc_lengths.items()),
            "doc_titles": list(index.doc_titles.items()),
            "num_docs": index.num_docs,
        }
        with compressor.stream_writer(block_path.open("wb")) as f:
            f.write(orjson.dumps(block_data))
        print(f"Bloco {block_num} guardado (último bloco, {len(partial_index):,} termos)")

    pool.close()
    pool.join()

    total_time = time.time() - start_time
    print(f"\n{block_num + 1} blocos criados em {total_time:.2f}s")
