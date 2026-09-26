"""
mini_ir.py
A minimal Information Retrieval (IR) system built from scratch, using only
Python's standard library (no numpy, no sklearn, no ML frameworks).

Implements:
  1. Text preprocessing (tokenize, lowercase, stopword removal)
  2. TF-IDF vector space model + cosine similarity
  3. BM25 ranking function
  4. Reciprocal Rank Fusion (RRF) to combine both into one hybrid ranking

Run directly to see a demo: `python mini_ir.py`
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "for", "with", "this", "that", "it", "as", "be",
}


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into alphanumeric tokens."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def preprocess(text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
    """Tokenize and strip stopwords (uses default STOPWORDS if none provided)."""
    sw = STOPWORDS if stopwords is None else stopwords
    return [t for t in tokenize(text) if t not in sw]


class IRIndex:
    """
    Builds document statistics (term frequencies, document frequencies,
    document lengths) over a corpus, and exposes TF-IDF and BM25 scoring
    against a query, plus RRF fusion of the two.
    """

    def __init__(
        self,
        documents: List[str],
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the Information Retrieval index with a collection of documents.

        Args:
            documents: List of raw text documents to index. Must not be empty.
            bm25_k1: BM25 term-frequency saturation constant (default: 1.5).
            bm25_b: BM25 document length normalization strength, 0=off, 1=full (default: 0.75).
            stopwords: Optional custom set of stopwords to filter during preprocessing.
        """
        if not documents:
            raise ValueError("Document corpus cannot be empty.")
        if bm25_k1 < 0:
            raise ValueError(f"bm25_k1 must be non-negative, got {bm25_k1}")
        if not (0.0 <= bm25_b <= 1.0):
            raise ValueError(f"bm25_b must be between 0.0 and 1.0, got {bm25_b}")

        self.documents: List[str] = list(documents)
        self.stopwords: Optional[Set[str]] = stopwords
        self.N: int = len(documents)
        self.doc_tokens: List[List[str]] = [preprocess(doc, self.stopwords) for doc in documents]
        self.doc_term_freqs: List[Counter] = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lengths: List[int] = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_length: float = sum(self.doc_lengths) / self.N

        self.k1: float = bm25_k1   # BM25 term-frequency saturation constant
        self.b: float = bm25_b      # BM25 length-normalization strength (0=off, 1=full)

        self.df: Dict[str, int] = self._document_frequencies()

        # Two different IDF variants: plain log(N/df) for TF-IDF, and the
        # smoothed Robertson/Sparck Jones variant for BM25 (stays non-negative).
        self.idf_tfidf: Dict[str, float] = {
            t: math.log(self.N / df) for t, df in self.df.items()
        }
        self.idf_bm25: Dict[str, float] = {
            t: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self.df.items()
        }

    def _document_frequencies(self) -> Dict[str, int]:
        """How many documents contain each term."""
        df: Dict[str, int] = defaultdict(int)
        for tf in self.doc_term_freqs:
            for term in tf:
                df[term] += 1
        return dict(df)

    # ---------------- TF-IDF + cosine similarity ----------------

    def _tfidf_vector(self, term_freqs: Counter, idf_table: Dict[str, float]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for term, freq in term_freqs.items():
            tf = 1.0 + math.log(freq)        # log-scaled term frequency
            idf = idf_table.get(term, 0.0)   # 0.0 for unseen (out-of-vocab) terms
            vec[term] = tf * idf
        return vec

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        common_terms = set(vec_a) & set(vec_b)
        if not common_terms:
            return 0.0
        dot = sum(vec_a[t] * vec_b[t] for t in common_terms)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_tfidf(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Rank documents using TF-IDF vectors and cosine similarity.

        Args:
            query: The search query string.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of (doc_id, score) tuples sorted by score in descending order.
        """
        if top_k <= 0:
            return []

        query_terms = Counter(preprocess(query, self.stopwords))
        query_vec = self._tfidf_vector(query_terms, self.idf_tfidf)

        scores: List[Tuple[int, float]] = []
        for doc_id, term_freqs in enumerate(self.doc_term_freqs):
            doc_vec = self._tfidf_vector(term_freqs, self.idf_tfidf)
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ---------------- BM25 ----------------

    def search_bm25(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Rank documents using the BM25 scoring algorithm with document length normalization.

        Args:
            query: The search query string.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of (doc_id, score) tuples sorted by score in descending order.
        """
        if top_k <= 0:
            return []

        query_terms = preprocess(query, self.stopwords)
        scores: List[Tuple[int, float]] = []

        for doc_id, term_freqs in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lengths[doc_id]
            score = 0.0
            for term in query_terms:
                if term not in term_freqs:
                    continue
                f = term_freqs[term]
                idf = self.idf_bm25.get(term, 0.0)
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avg_doc_length if self.avg_doc_length > 0 else 1.0)
                )
                score += idf * (numerator / denominator)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ---------------- Hybrid fusion (Reciprocal Rank Fusion) ----------------

    def search_hybrid_rrf(
        self, query: str, top_k: int = 5, k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Fuse TF-IDF and BM25 rankings using Reciprocal Rank Fusion (RRF):
        score(d) = sum(1 / (k + rank)).

        Same mechanic used to fuse dense (vector) + sparse (BM25) retrieval
        in production hybrid search -- here applied to two lexical methods
        to keep everything runnable without an embedding model.

        Args:
            query: The search query string.
            top_k: Maximum number of fused results to return.
            k: Smoothing constant to control impact of top-ranked items (default: 60).

        Returns:
            List of (doc_id, score) tuples sorted by fused RRF score in descending order.
        """
        if top_k <= 0:
            return []

        tfidf_ranked = self.search_tfidf(query, top_k=self.N)
        bm25_ranked = self.search_bm25(query, top_k=self.N)

        rrf_scores: Dict[int, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(tfidf_ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
        for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]


# ---------------------------- CLI & Demo ----------------------------

DEMO_CORPUS = [
    "The cat sat on the mat and looked at the dog.",
    "Dogs are loyal animals often kept as pets.",
    "Python is a popular programming language for data science.",
    "Machine learning models can be trained using Python libraries.",
    "Cats and dogs are the most common household pets.",
    "Data science relies heavily on statistics and programming.",
]


def run_demo(query: str = "python programming for machine learning", top_k: int = 3) -> None:
    """Run an interactive demonstration of TF-IDF, BM25, and Hybrid RRF."""
    index = IRIndex(DEMO_CORPUS)
    print(f"Query: {query!r}\n")

    print("-- TF-IDF ranking --")
    for doc_id, score in index.search_tfidf(query, top_k=top_k):
        print(f"  [{score:.4f}] {DEMO_CORPUS[doc_id]}")

    print("\n-- BM25 ranking --")
    for doc_id, score in index.search_bm25(query, top_k=top_k):
        print(f"  [{score:.4f}] {DEMO_CORPUS[doc_id]}")

    print("\n-- Hybrid (RRF fusion) ranking --")
    for doc_id, score in index.search_hybrid_rrf(query, top_k=top_k):
        print(f"  [{score:.4f}] {DEMO_CORPUS[doc_id]}")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for mini-ir."""
    parser = argparse.ArgumentParser(
        prog="mini-ir",
        description="A minimal, zero-dependency Information Retrieval engine in pure Python.",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default="python programming for machine learning",
        help="Search query to run against corpus.",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Number of ranked documents to return (default: 3).",
    )
    parser.add_argument(
        "-m", "--method",
        choices=["all", "tfidf", "bm25", "rrf"],
        default="all",
        help="Ranking method to evaluate (default: all).",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Optional path to a newline-delimited text file containing documents.",
    )

    args = parser.parse_args(argv)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                corpus = [line.strip() for line in f if line.strip()]
            if not corpus:
                print(f"Error: file '{args.file}' contained no documents.", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
            return 1
    else:
        corpus = DEMO_CORPUS

    index = IRIndex(corpus)
    print(f"Query: {args.query!r}\n")

    if args.method in ("all", "tfidf"):
        print("-- TF-IDF ranking --")
        for doc_id, score in index.search_tfidf(args.query, top_k=args.top_k):
            print(f"  [{score:.4f}] {corpus[doc_id]}")
        if args.method == "all":
            print()

    if args.method in ("all", "bm25"):
        print("-- BM25 ranking --")
        for doc_id, score in index.search_bm25(args.query, top_k=args.top_k):
            print(f"  [{score:.4f}] {corpus[doc_id]}")
        if args.method == "all":
            print()

    if args.method in ("all", "rrf"):
        print("-- Hybrid (RRF fusion) ranking --")
        for doc_id, score in index.search_hybrid_rrf(args.query, top_k=args.top_k):
            print(f"  [{score:.4f}] {corpus[doc_id]}")

    return 0


if __name__ == "__main__":
    # If no arguments provided, run default demo seamlessly
    if len(sys.argv) == 1:
        run_demo()
    else:
        sys.exit(main())
