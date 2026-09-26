"""
Unit tests for mini_ir using Python's standard library unittest.
Zero external test dependencies required.
"""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout

import mini_ir
from mini_ir import IRIndex, preprocess, tokenize


class TestTokenizationAndPreprocessing(unittest.TestCase):
    def test_tokenize_basic(self):
        text = "Hello, World! 123."
        tokens = tokenize(text)
        self.assertEqual(tokens, ["hello", "world", "123"])

    def test_tokenize_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   !@#$%^&*()   "), [])

    def test_preprocess_default_stopwords(self):
        text = "The cat is on the mat and it was fast"
        tokens = preprocess(text)
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("on", tokens)
        self.assertNotIn("and", tokens)
        self.assertNotIn("it", tokens)
        self.assertNotIn("was", tokens)
        self.assertEqual(tokens, ["cat", "mat", "fast"])

    def test_preprocess_custom_stopwords(self):
        text = "The quick brown fox jumps"
        custom_sw = {"quick", "fox"}
        tokens = preprocess(text, stopwords=custom_sw)
        # "the" is kept because default stopwords were overridden
        self.assertEqual(tokens, ["the", "brown", "jumps"])


class TestIRIndexInitialization(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Apple makes smartphones and computers.",
            "Orange is a citrus fruit and computers use silicon.",
            "Banana and orange are nutritious fruits.",
        ]

    def test_empty_corpus_raises_error(self):
        with self.assertRaises(ValueError):
            IRIndex([])

    def test_invalid_parameters_raise_error(self):
        with self.assertRaises(ValueError):
            IRIndex(self.corpus, bm25_k1=-0.5)

        with self.assertRaises(ValueError):
            IRIndex(self.corpus, bm25_b=-0.1)

        with self.assertRaises(ValueError):
            IRIndex(self.corpus, bm25_b=1.2)

    def test_document_frequencies(self):
        index = IRIndex(self.corpus)
        # "computers" appears in doc 0 and doc 1
        self.assertEqual(index.df["computers"], 2)
        # "apple" appears in doc 0 only
        self.assertEqual(index.df["apple"], 1)
        # "orange" appears in doc 1 and doc 2
        self.assertEqual(index.df["orange"], 2)

    def test_bm25_idf_is_positive(self):
        index = IRIndex(self.corpus)
        for term, idf in index.idf_bm25.items():
            self.assertGreater(idf, 0.0, f"BM25 IDF for {term} should be strictly positive")

    def test_tfidf_idf_formula(self):
        index = IRIndex(self.corpus)
        for term, df in index.df.items():
            expected = math.log(len(self.corpus) / df)
            self.assertAlmostEqual(index.idf_tfidf[term], expected)


class TestSearchTFIDF(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Deep learning with neural networks and backpropagation.",
            "Information retrieval using inverted index and vector space models.",
            "Convolutional neural networks for computer vision.",
        ]
        self.index = IRIndex(self.corpus)

    def test_search_relevance(self):
        results = self.index.search_tfidf("neural networks", top_k=3)
        self.assertEqual(len(results), 3)
        top_doc_id, top_score = results[0]
        # Docs 0 and 2 contain "neural networks"
        self.assertIn(top_doc_id, [0, 2])
        self.assertGreater(top_score, 0.0)

    def test_cosine_similarity_bounds(self):
        results = self.index.search_tfidf("neural", top_k=len(self.corpus))
        for _, score in results:
            self.assertTrue(0.0 <= score <= 1.0 + 1e-9)

    def test_unseen_query(self):
        results = self.index.search_tfidf("quantum physics aerospace", top_k=3)
        for _, score in results:
            self.assertEqual(score, 0.0)

    def test_empty_query(self):
        results = self.index.search_tfidf("", top_k=3)
        for _, score in results:
            self.assertEqual(score, 0.0)

    def test_top_k_zero_or_negative(self):
        self.assertEqual(self.index.search_tfidf("neural", top_k=0), [])
        self.assertEqual(self.index.search_tfidf("neural", top_k=-1), [])


class TestSearchBM25(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning algorithms optimize predictive loss functions.",
            "Fast search engine indexing algorithms.",
        ]
        self.index = IRIndex(self.corpus)

    def test_search_relevance(self):
        results = self.index.search_bm25("search engine algorithms", top_k=3)
        self.assertEqual(results[0][0], 2)  # Doc 2 has both search, engine, algorithms
        self.assertGreater(results[0][1], results[1][1])

    def test_top_k_bounds(self):
        results = self.index.search_bm25("algorithms", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(self.index.search_bm25("algorithms", top_k=0), [])

    def test_length_normalization(self):
        # Doc 0 is short, Doc 1 is long with lots of filler words
        corpus = [
            "database systems",
            "database systems with advanced distributed architecture and lots of redundant filler tokens",
        ]
        # With b=0.75, short document should score higher for identical matched terms
        idx = IRIndex(corpus, bm25_b=0.75)
        results = idx.search_bm25("database systems")
        self.assertEqual(results[0][0], 0)

    def test_term_frequency_saturation(self):
        corpus = [
            "python",
            "python python python python python",
        ]
        idx = IRIndex(corpus, bm25_k1=1.5, bm25_b=0.0)
        res = idx.search_bm25("python")
        score_1 = dict(res)[0]
        score_5 = dict(res)[1]
        # Due to saturation f * (k1 + 1) / (f + k1), 5x frequency should NOT result in 5x score
        self.assertLess(score_5, score_1 * 2.5)


class TestSearchHybridRRF(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Python programming and machine learning libraries.",
            "Vector databases enable semantic similarity search.",
            "Full text search engines use BM25 ranking algorithms.",
        ]
        self.index = IRIndex(self.corpus)

    def test_rrf_scoring(self):
        results = self.index.search_hybrid_rrf("python machine learning", top_k=3, k=60)
        self.assertEqual(results[0][0], 0)  # Doc 0 ranks 1st in both
        # Rank 1 in both: score should be 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032786
        expected_score = (1.0 / 61.0) + (1.0 / 61.0)
        self.assertAlmostEqual(results[0][1], expected_score, places=5)

    def test_top_k_handling(self):
        self.assertEqual(self.index.search_hybrid_rrf("python", top_k=0), [])
        results = self.index.search_hybrid_rrf("python", top_k=1)
        self.assertEqual(len(results), 1)


class TestCLIAndDemo(unittest.TestCase):
    def test_run_demo_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            mini_ir.run_demo()
        output = buf.getvalue()
        self.assertIn("TF-IDF ranking", output)
        self.assertIn("BM25 ranking", output)
        self.assertIn("Hybrid (RRF fusion) ranking", output)

    def test_main_cli_with_arguments(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = mini_ir.main(["-q", "programming", "-m", "tfidf", "-k", "2"])
        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("TF-IDF ranking", output)

    def test_main_cli_with_custom_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            f.write("First custom document line.\nSecond custom document line.\n")
            f_path = f.name

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = mini_ir.main(["-f", f_path, "-q", "custom document", "-m", "bm25"])
        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("BM25 ranking", output)
class TestStandaloneRRF(unittest.TestCase):
    def test_standalone_rrf(self):
        dense_results = [(0, 0.95), (1, 0.85), (2, 0.70)]
        bm25_results = [(1, 4.2), (0, 3.1), (3, 2.0)]
        fused = mini_ir.reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_k=2)
        self.assertEqual(len(fused), 2)
        # Doc 0: 1/61 + 1/62 ≈ 0.016393 + 0.016129 = 0.032522
        # Doc 1: 1/62 + 1/61 = 0.032522 (tie)
        # Both doc 0 and 1 should be the top 2
        fused_ids = [doc_id for doc_id, _ in fused]
        self.assertIn(0, fused_ids)
        self.assertIn(1, fused_ids)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_json(self):
        corpus = ["First document text", "Second document text"]
        idx1 = IRIndex(corpus, bm25_k1=1.2, bm25_b=0.8)

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            f_path = f.name

        idx1.save_json(f_path)
        idx2 = IRIndex.load_json(f_path)

        self.assertEqual(idx2.documents, corpus)
        self.assertEqual(idx2.k1, 1.2)
        self.assertEqual(idx2.b, 0.8)
        self.assertEqual(idx1.search_bm25("document"), idx2.search_bm25("document"))


class TestCustomTokenizer(unittest.TestCase):
    def test_custom_tokenizer(self):
        # Custom tokenizer that splits by whitespace only
        corpus = ["foo-bar baz", "test-case foo"]
        custom_tok = lambda s: s.split()
        idx = IRIndex(corpus, tokenizer=custom_tok)
        res = idx.search_bm25("foo-bar")
        self.assertEqual(res[0][0], 0)


if __name__ == "__main__":
    unittest.main()

