"""Rankings TF-IDF/coseno y BM25."""
import math
from collections import Counter, defaultdict

from .inverted_index import InvertedIndex
from .preprocess import preprocess


def _rank(index: InvertedIndex, scores: dict[int, float], top_k: int) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], index.doc_ids[item[0]]))
    return [(index.doc_ids[position], float(score)) for position, score in ranked[:top_k]]


def tfidf_search(index: InvertedIndex, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    query_tf = Counter(term for term in preprocess(query) if term in index.postings)
    query_weights = {term: frequency * index.tfidf_idf(term) for term, frequency in query_tf.items()}
    query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
    if query_norm == 0:
        return []
    dot_products: defaultdict[int, float] = defaultdict(float)
    for term, query_weight in query_weights.items():
        idf = index.tfidf_idf(term)
        for position, frequency in index.postings[term].items():
            dot_products[position] += query_weight * frequency * idf
    scores = {
        position: dot / (query_norm * index.tfidf_norm[position])
        for position, dot in dot_products.items()
        if index.tfidf_norm[position] > 0
    }
    return _rank(index, scores, top_k)


def bm25_search(
    index: InvertedIndex, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75
) -> list[tuple[str, float]]:
    scores: defaultdict[int, float] = defaultdict(float)
    for term, query_frequency in Counter(preprocess(query)).items():
        if term not in index.postings:
            continue
        idf = index.bm25_idf(term)
        for position, frequency in index.postings[term].items():
            length_normalization = frequency + k1 * (
                1 - b + b * index.doc_len[position] / index.average_document_length
            )
            scores[position] += (
                query_frequency * idf * frequency * (k1 + 1) / length_normalization
            )
    return _rank(index, scores, top_k)

