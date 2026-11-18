"""Construye el indice y calcula P@10, R@10 y NDCG@10."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate
from src.inverted_index import InvertedIndex
from src.models import bm25_search, tfidf_search


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    data = ROOT / "data"
    artifacts = ROOT / "artifacts"
    documents = {}
    with (data / "corpus.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            documents[row["_id"]] = f"{row.get('title', '')} {row['text']}".strip()
    queries, qrels = load_json(data / "queries.json"), load_json(data / "qrels.json")
    index = InvertedIndex().build(documents)
    index.save(artifacts / "index.pkl")

    models = {
        "tfidf": lambda query, k: tfidf_search(index, query, k),
        "bm25": lambda query, k: bm25_search(index, query, k),
    }
    metrics = {"k": 10, "models": {name: evaluate(fn, queries, qrels) for name, fn in models.items()}}
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Indice: {index.n_documents} documentos, {len(index.postings)} terminos")
    for name, row in metrics["models"].items():
        print(
            f"{name.upper()}: P@10={row['mean_precision@10']:.4f} | "
            f"R@10={row['mean_recall@10']:.4f} | NDCG@10={row['mean_ndcg@10']:.4f}"
        )


if __name__ == "__main__":
    main()

