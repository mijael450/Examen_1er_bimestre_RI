"""Descarga un corpus BEIR y conserva sus identificadores originales."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


SUPPORTED = {"scifact", "nfcorpus"}
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(SUPPORTED), default="scifact")
    args = parser.parse_args()
    output = ROOT / "data"
    output.mkdir(exist_ok=True)

    dataset_id = f"BeIR/{args.corpus}"
    corpus_rows = load_dataset(dataset_id, "corpus", split="corpus")
    query_rows = load_dataset(dataset_id, "queries", split="queries")
    qrel_rows = load_dataset(f"{dataset_id}-qrels", split="test")

    documents = {
        str(row["_id"]): {
            "_id": str(row["_id"]),
            "title": (row.get("title") or "").strip(),
            "text": (row.get("text") or "").strip(),
        }
        for row in corpus_rows
        if (row.get("text") or "").strip()
    }
    all_queries = {str(row["_id"]): row["text"] for row in query_rows}
    qrels: dict[str, dict[str, int]] = {}
    for row in qrel_rows:
        query_id, document_id, score = (
            str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        )
        if score > 0 and query_id in all_queries and document_id in documents:
            qrels.setdefault(query_id, {})[document_id] = score
    queries = {query_id: all_queries[query_id] for query_id in qrels}

    with (output / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for document in documents.values():
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    (output / "queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "qrels.json").write_text(json.dumps(qrels, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "documents": len(documents),
                "test_queries": len(queries),
                "qrel_pairs": sum(len(items) for items in qrels.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Corpus: {args.corpus}")
    print(f"Documentos: {len(documents)}")
    print(f"Consultas test: {len(queries)}")
    print(f"Pares relevantes: {sum(len(items) for items in qrels.values())}")


if __name__ == "__main__":
    main()

