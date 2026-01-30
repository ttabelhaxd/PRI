from pathlib import Path

import pyarrow.dataset as ds

from sapien.core.model import Document


class CorpusLoader:
    """Lê o corpus Wikipedia em modo streaming, eficiente em memória."""

    def __init__(self, file_name: str = "ptwiki-small-50000.arrow", batch_size: int = 10000):
        self.data_path = Path(__file__).resolve().parents[3] / "data" / file_name
        self.batch_size = batch_size

    def __iter__(self):
        """Gera objetos Document de forma incremental (sem carregar tudo na RAM)."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {self.data_path}")

        dataset = ds.dataset(str(self.data_path), format="arrow")

        doc_id = 0
        # Leitura em blocos
        for batch in dataset.to_batches(batch_size=self.batch_size):
            rows = batch.to_pylist()
            for row in rows:
                if row.get("redirect", False):
                    continue

                text = row.get("text", "").strip()
                title = row.get("title", "").strip()

                if not text or not title:
                    continue

                yield Document(id=doc_id, title=title, content=text)
                doc_id += 1
