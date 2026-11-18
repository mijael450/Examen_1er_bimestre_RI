import json
import tempfile
import unittest
from pathlib import Path

import api
from src.evaluate import ndcg_at_k, precision_at_k, recall_at_k
from src.inverted_index import InvertedIndex
from src.models import bm25_search, tfidf_search


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex().build(
            {"doc-a": "heart disease treatment", "doc-b": "cancer cell treatment", "doc-c": "astronomy stars"}
        )

    def test_models_return_relevant_document_first(self):
        for search in (tfidf_search, bm25_search):
            result = search(self.index, "heart disease", 2)
            self.assertEqual("doc-a", result[0][0])
            self.assertEqual(len(result), len({document_id for document_id, _ in result}))
            self.assertTrue(all(result[i][1] >= result[i + 1][1] for i in range(len(result) - 1)))

    def test_unknown_terms_return_empty_ranking(self):
        self.assertEqual([], tfidf_search(self.index, "zzzzunknown", 10))
        self.assertEqual([], bm25_search(self.index, "zzzzunknown", 10))

    def test_index_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pkl"
            self.index.save(path)
            self.assertEqual(self.index.doc_ids, InvertedIndex.load(path).doc_ids)

    def test_metrics(self):
        ranked = ["a", "x", "b"]
        self.assertAlmostEqual(0.2, precision_at_k(ranked, {"a", "b"}, 10))
        self.assertEqual(1.0, recall_at_k(ranked, {"a", "b"}, 10))
        self.assertGreater(ndcg_at_k(ranked, {"a": 2, "b": 1}, 3), 0.0)


class ApiTests(unittest.TestCase):
    def test_health_search_and_validation(self):
        api.load_resources()
        self.assertEqual(5183, api.health()["documents_indexed"])

        for model in ("tfidf", "bm25"):
            payload = api.search(
                api.SearchRequest(query="vitamin D cancer risk", model=model, top_k=5)
            )
            self.assertEqual(model, payload["model"])
            self.assertEqual(5, len(payload["results"]))
            self.assertEqual(list(range(1, 6)), [row["rank"] for row in payload["results"]])
            self.assertEqual(
                len(payload["results"]),
                len({row["document_id"] for row in payload["results"]}),
            )
            self.assertTrue(
                all(
                    payload["results"][i]["score"] >= payload["results"][i + 1]["score"]
                    for i in range(4)
                )
            )


if __name__ == "__main__":
    unittest.main()

