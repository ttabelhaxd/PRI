import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NeuralReranker:
    """Neural Re-ranking using the cross-encoder `unicamp-dl/mMiniLM-L6-v2-pt-msmarco-v1`.
    This model takes (query, document_text) pairs and returns a relevance score.
    """

    def __init__(
        self, model_name: str = "unicamp-dl/mMiniLM-L6-v2-pt-msmarco-v1", device: str | None = None
    ):
        print(f"[RERANKER] Loading model '{model_name}' ...")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            print("[RERANKER] GPU loaded:", torch.cuda.get_device_name(0))
        else:
            print("[RERANKER] GPU not available. Using CPU.")

        print(f"[RERANKER] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

        print("[RERANKER] Model loaded successfully.")

    def score_pair(self, query: str, document: str) -> float:
        """Compute the relevance score for a single (query, document) pair."""

        encoded = self.tokenizer(
            query, document, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            score = output.logits.squeeze().item()

        return float(score)

    def rerank(self, query: str, docs: list[dict], batch_size: int = 16) -> list[dict]:
        """Efficient reranking using batching on GPU."""

        if not docs:
            return []

        print(f"[RERANKER] Reranking {len(docs)} candidates (batched)...")

        scores = []
        texts = [doc.get("text", "") for doc in docs]

        # Process documents in batches
        for i in range(0, len(docs), batch_size):
            batch_docs = texts[i : i + batch_size]

            encoded = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)
                batch_scores = output.logits.squeeze().tolist()

            # Guarda scores no array global
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            scores.extend(batch_scores)

        # Juntar docs e scores → ordenar
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        print("[RERANKER] Done with batching.")
        return [d for (d, _) in scored_docs]
