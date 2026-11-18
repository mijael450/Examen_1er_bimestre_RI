"""Servicio REST local para TF-IDF y BM25."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inverted_index import InvertedIndex
from src.models import bm25_search, tfidf_search


ROOT = Path(__file__).resolve().parent
STATE: dict = {}


def load_resources() -> None:
    index_path = ROOT / "artifacts" / "index.pkl"
    corpus_path = ROOT / "data" / "corpus.jsonl"
    if not index_path.exists() or not corpus_path.exists():
        raise RuntimeError("Faltan datos/indice. Ejecuta prepare_data.py y build_index.py.")
    STATE["index"] = InvertedIndex.load(index_path)
    with corpus_path.open(encoding="utf-8") as stream:
        STATE["documents"] = {row["_id"]: row for row in map(json.loads, stream)}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_resources()
    yield
    STATE.clear()


app = FastAPI(title="Recuperacion de Informacion BEIR", version="1.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    model: str
    top_k: int = Field(default=10, ge=1, le=100)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents_indexed": STATE["index"].n_documents}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    model = request.model.lower().strip()
    functions = {"tfidf": tfidf_search, "bm25": bm25_search}
    if model not in functions:
        raise HTTPException(status_code=422, detail="model debe ser 'tfidf' o 'bm25'")
    ranking = functions[model](STATE["index"], request.query, request.top_k)
    results = []
    for rank, (document_id, score) in enumerate(ranking, start=1):
        document = STATE["documents"][document_id]
        text = " ".join(document["text"].split())
        results.append(
            {
                "rank": rank,
                "document_id": document_id,
                "score": round(score, 8),
                "title": document.get("title", ""),
                "snippet": text[:300] + ("..." if len(text) > 300 else ""),
            }
        )
    return {"model": model, "results": results}

