import gc
import heapq
import io
import shutil
from collections import defaultdict
from pathlib import Path

import orjson
import zstandard as zstd

gc.disable()


def merge_indexes_for_batch(block_files, output_file, is_partial_merge=True):
    """Merge de índices (compatível com blocos monolíticos e streaming)."""
    print(f"\nA fundir {len(block_files)} blocos → {output_file.name} ...")

    if not block_files:
        print("Nenhum bloco para fundir.")
        return

    total_docs = 0
    heap = []

    class StreamingBlockIterator:
        """Iterador que deteta automaticamente o formato do bloco (monolítico ou streaming)."""

        def __init__(self, block_path, block_id, is_partial):
            self.block_path = block_path
            self.block_id = block_id
            self.is_partial = is_partial
            self.reader = None
            self.metadata = None
            self.exhausted = False
            self.index_iter = None  # usado para blocos monolíticos

        def _init_stream(self):
            dctx = zstd.ZstdDecompressor()
            raw_file = self.block_path.open("rb")

            # tenta leitura linha a linha (streaming)
            stream = dctx.stream_reader(raw_file)
            self.reader = io.TextIOWrapper(
                io.BufferedReader(stream, buffer_size=8192 * 16), encoding="utf-8"
            )

            header_line = self.reader.readline()
            if not header_line:
                self.exhausted = True
                return

            try:
                maybe_data = orjson.loads(header_line.strip())
                # formato monolítico (indexer original)
                if "index" in maybe_data:
                    self._init_from_monolithic(maybe_data)
                    try:
                        self.reader.close()
                    except Exception:
                        pass
                    self.reader = None
                    return
                else:
                    # formato streaming (merge intermediário)
                    self.metadata = maybe_data
                    _ = self.reader.readline()
            except Exception:
                # fallback: tenta formato monolítico completo
                raw_file.seek(0)
                raw = dctx.decompress(raw_file.read())
                try:
                    data = orjson.loads(raw)
                    self._init_from_monolithic(data)
                    try:
                        self.reader.close()
                    except Exception:
                        pass
                    self.reader = None
                except Exception:
                    self.exhausted = True

        def _init_from_monolithic(self, data):
            """Inicializa iterador a partir de um bloco único (formato do indexer)."""
            doc_lengths = data.get("doc_lengths", {})
            doc_titles = data.get("doc_titles", {})
            if isinstance(doc_lengths, list):
                doc_lengths = dict(doc_lengths)
            if isinstance(doc_titles, list):
                doc_titles = dict(doc_titles)

            self.metadata = {
                "doc_lengths": doc_lengths,
                "doc_titles": doc_titles,
                "num_docs": data.get("num_docs", 0),
            }

            index_items = data.get("index", [])
            if isinstance(index_items, dict):
                index_items = sorted(index_items.items())
            else:
                index_items = sorted(index_items)
            self.index_iter = iter(index_items)

        def get_metadata(self):
            if self.metadata is None:
                self._init_stream()
            return self.metadata or {}

        def read_next(self):
            """Lê o próximo termo."""
            if self.metadata is None and self.reader is None and self.index_iter is None:
                self._init_stream()

            # formato monolítico
            if self.index_iter is not None:
                try:
                    return next(self.index_iter)
                except StopIteration:
                    return None

            # formato streaming
            if self.reader is None or self.exhausted:
                return None

            try:
                while True:
                    line = self.reader.readline()
                    if not line:
                        self.exhausted = True
                        return None
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        term_entry = orjson.loads(line)
                        for term, postings in term_entry.items():
                            return term, postings
                    except Exception:
                        continue
            except Exception:
                self.exhausted = True
                return None

        def close(self):
            if self.reader:
                try:
                    self.reader.close()
                except Exception:
                    pass

    # Inicializa iteradores
    iterators = []
    for i, block_path in enumerate(block_files):
        it = StreamingBlockIterator(block_path, i, is_partial_merge)
        meta = it.get_metadata()
        total_docs += meta.get("num_docs", 0)
        first = it.read_next()
        if first:
            term, postings = first
            heapq.heappush(heap, (term, i, postings, it))
            iterators.append(it)

    print(f"{len(iterators)} blocos abertos em streaming ({total_docs:,} docs totais)")

    temp_file = output_file.parent / f".tmp_{output_file.name}"
    current_term = None
    current_postings = defaultdict(list)
    processed_terms = 0

    cctx = zstd.ZstdCompressor(level=0)

    with open(temp_file, "wb") as f_out:
        with cctx.stream_writer(f_out, closefd=False) as writer:
            # --- Coletar todos os metadados (títulos e comprimentos) ---
            merged_doc_lengths = {}
            merged_doc_titles = {}

            for it in iterators:
                meta = it.get_metadata()
                merged_doc_lengths.update(meta.get("doc_lengths", {}))
                merged_doc_titles.update(meta.get("doc_titles", {}))

            header = {
                "num_docs": total_docs,
                "doc_lengths": merged_doc_lengths,
                "doc_titles": merged_doc_titles,
            }

            writer.write(orjson.dumps(header))
            writer.write(b"\n---INDEX---\n")

            while heap:
                term, block_id, postings, it = heapq.heappop(heap)

                if term != current_term:
                    if current_term is not None:
                        merged_postings = [
                            [doc_id, sorted(pos)]
                            for doc_id, pos in sorted(current_postings.items())
                        ]
                        writer.write(orjson.dumps({current_term: merged_postings}))
                        writer.write(b"\n")
                        current_postings.clear()
                        processed_terms += 1

                        if processed_terms % 100000 == 0:
                            print(f"  • {processed_terms:,} termos fundidos...")
                            writer.flush()
                            gc.collect()

                    current_term = term

                for doc_id, positions in postings:
                    if isinstance(positions, list):
                        current_postings[doc_id].extend(positions)
                    else:
                        current_postings[doc_id].append(positions)

                next_item = it.read_next()
                del postings
                if next_item:
                    next_term, next_postings = next_item
                    heapq.heappush(heap, (next_term, block_id, next_postings, it))

            if current_term and current_postings:
                merged_postings = [
                    [doc_id, sorted(pos)] for doc_id, pos in sorted(current_postings.items())
                ]
                writer.write(orjson.dumps({current_term: merged_postings}))
                writer.write(b"\n")
                processed_terms += 1

    for it in iterators:
        it.close()

    shutil.move(str(temp_file), str(output_file))
    gc.collect()

    print(f"Merge concluído: {processed_terms:,} termos únicos — {total_docs:,} docs totais")
    return processed_terms


def merge_in_batches(index_dir, output_dir, batch_size=5):
    """Merge hierárquico otimizado para grandes coleções."""
    output_dir.mkdir(parents=True, exist_ok=True)
    block_files = sorted(index_dir.glob("index_block_*.zst"))

    if not block_files:
        print("Nenhum bloco encontrado.")
        return

    print(f"Total: {len(block_files)} blocos no diretório de índices")

    # Fase 1 — merges em batches mais pequenos
    batches = [block_files[i : i + batch_size] for i in range(0, len(block_files), batch_size)]
    partial_outputs = []

    print(f"Fase 1: {len(batches)} batches de ~{batch_size} blocos cada")

    for i, batch in enumerate(batches, 1):
        partial_output = output_dir / f"partial_merge_{i}.zst"
        print(f"\n[Batch {i}/{len(batches)}]")
        merge_indexes_for_batch(batch, partial_output, is_partial_merge=False)
        partial_outputs.append(partial_output)
        gc.collect()

    # Fase 2 — merge final (streaming)
    print(f"\nFase 2: merge final de {len(partial_outputs)} merges parciais")
    final_output = output_dir / "merged_index_final.zst"
    merge_indexes_for_batch(partial_outputs, final_output, is_partial_merge=True)

    # Limpeza
    print("\n A remover ficheiros temporários...")
    for partial in partial_outputs:
        try:
            partial.unlink()
        except Exception:
            pass

    print(f"\n Concluído: {final_output}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    index_dir = base_dir / "data" / "indexes_big_data"
    output_dir = index_dir / "merged_batched"

    merge_in_batches(index_dir, output_dir, batch_size=5)
