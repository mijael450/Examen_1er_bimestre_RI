"""Indice invertido en memoria, serializable con pickle."""
import math
import pickle
from collections import Counter
from pathlib import Path

from .preprocess import preprocess


class InvertedIndex:
    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_len: list[int] = []
        self.tfidf_norm: list[float] = []
        self.n_documents = 0
        self.average_document_length = 0.0

    def build(self, documents: dict[str, str]) -> "InvertedIndex":
        for position, (document_id, text) in enumerate(documents.items()):
            counts = Counter(preprocess(text))
            self.doc_ids.append(str(document_id))
            self.doc_len.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings.setdefault(term, {})[position] = frequency
        self.n_documents = len(self.doc_ids)
        self.average_document_length = sum(self.doc_len) / max(1, self.n_documents)
        self._compute_tfidf_norms()
        return self

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def tfidf_idf(self, term: str) -> float:
        frequency = self.document_frequency(term)
        return math.log(self.n_documents / frequency) if frequency else 0.0

    def bm25_idf(self, term: str) -> float:
        frequency = self.document_frequency(term)
        return math.log(1 + (self.n_documents - frequency + 0.5) / (frequency + 0.5))

    def _compute_tfidf_norms(self) -> None:
        squared = [0.0] * self.n_documents
        for term, posting in self.postings.items():
            idf = self.tfidf_idf(term)
            for position, frequency in posting.items():
                squared[position] += (frequency * idf) ** 2
        self.tfidf_norm = [math.sqrt(value) for value in squared]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path: Path) -> "InvertedIndex":
        with path.open("rb") as stream:
            return pickle.load(stream)

