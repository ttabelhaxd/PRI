"""Search endpoints — integra o motor BM25 real com FastAPI."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from sapien.core.model import Document
from sapien.core.searcher import Searcher
from sapien.core.tokenizer import Tokenizer
from sapien.entrypoints.api.model import SearchResponse

INDEX_PATH = (
    Path(__file__).resolve().parents[5] / "data" / "indexes" / "merged" / "merged_index_final.zst"
)

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"Índice não encontrado em: {INDEX_PATH}")

tokenizer = Tokenizer(
    lowercase=True,
    remove_punctuation=True,
    remove_accents_flag=True,
    remove_stopwords=True,
    use_stemming=True,
    min_length=3,
    keep_numbers=False,
    keep_alphanum=False,
)

# --- Carregar índice (única vez) ---
searcher = Searcher(INDEX_PATH, tokenizer)

# --- Router principal ---
router = APIRouter(tags=["search engine"])


# Pesquisa normal
@router.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="Texto a pesquisar (query)"),
    num_results: int = Query(10, description="Número máximo de resultados a devolver"),
) -> SearchResponse:
    """Pesquisa documentos relevantes usando BM25."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="A query não pode estar vazia.")

    ranked = searcher.search(query, top_k=num_results)
    results = []

    for doc_id, score in ranked:
        title = searcher.doc_titles.get(str(doc_id), f"Documento {doc_id}")
        snippet = searcher.get_snippet(doc_id)
        snippet_text = snippet if snippet else "Snippet não disponível."
        results.append(
            Document(
                id=int(doc_id),
                title=title,
                content=f"{snippet_text}\n\n(Relevância BM25: {score:.4f})"
            )
        )

    return SearchResponse(results=results)


# Pesquisa de documentos semelhantes (on-demand)
@router.get("/search_like", response_model=SearchResponse)
def search_like(
    doc_id: int = Query(..., description="ID do documento base"),
    num_results: int = Query(10, description="Número de documentos semelhantes a devolver"),
) -> SearchResponse:
    """Pesquisa documentos semelhantes a outro documento.
    Usa o índice direto leve (doc_terms) para gerar uma pseudo-query instantânea.
    """

    str_id = str(doc_id)

    if str_id not in searcher.doc_titles:
        raise HTTPException(status_code=404, detail=f"Documento {doc_id} não encontrado no índice.")

    # Obter os termos diretamente do índice direto
    terms_in_doc = list(searcher.doc_terms.get(str_id, []))[:30]
    if not terms_in_doc:
        raise HTTPException(
            status_code=404, detail=f"O documento {doc_id} não tem termos indexados."
        )

    print(f"{len(terms_in_doc)} termos encontrados → a gerar pseudo-query ...")

    pseudo_query = " ".join(terms_in_doc)
    ranked = searcher.search(pseudo_query, top_k=num_results + 1)

    # Remove o próprio documento dos resultados
    ranked = [(d, s) for (d, s) in ranked if d != str_id][:num_results]

    results = []
    for d, score in ranked:
        title = searcher.doc_titles.get(str(d), f"Documento {d}")
        snippet = searcher.get_snippet(d)
        snippet_text = snippet if snippet else "Snippet não disponível."
        results.append(
            Document(
                id=int(d),
                title=title,
                content=f"{snippet_text}\n\n(Semelhança BM25: {score:.4f})"
            )
        )

    return SearchResponse(results=results)
