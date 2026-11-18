"""Metricas solicitadas para qrels binarios o graduados."""
import math
from collections.abc import Callable


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(document_id in relevant for document_id in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(document_id in relevant for document_id in ranked[:k]) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))

    actual = dcg([int(judgments.get(document_id, 0)) for document_id in ranked[:k]])
    ideal = dcg(sorted((int(score) for score in judgments.values()), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluate(
    search: Callable[[str, int], list[tuple[str, float]]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict:
    per_query = {}
    for query_id, query in queries.items():
        judgments = qrels[query_id]
        relevant = {document_id for document_id, score in judgments.items() if int(score) > 0}
        ranked = [document_id for document_id, _ in search(query, k)]
        per_query[query_id] = {
            f"precision@{k}": precision_at_k(ranked, relevant, k),
            f"recall@{k}": recall_at_k(ranked, relevant, k),
            f"ndcg@{k}": ndcg_at_k(ranked, judgments, k),
        }
    count = max(1, len(per_query))
    return {
        "queries_evaluated": len(per_query),
        f"mean_precision@{k}": sum(row[f"precision@{k}"] for row in per_query.values()) / count,
        f"mean_recall@{k}": sum(row[f"recall@{k}"] for row in per_query.values()) / count,
        f"mean_ndcg@{k}": sum(row[f"ndcg@{k}"] for row in per_query.values()) / count,
        "per_query": per_query,
    }

