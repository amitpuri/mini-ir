"""
Quickstart Example for mini-ir.
Demonstrates:
  1. Indexing a document corpus
  2. TF-IDF vector space search with cosine similarity
  3. BM25 search with term saturation & document length normalization
  4. Hybrid search using Reciprocal Rank Fusion (RRF)
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_ir import IRIndex


def main():
    corpus = [
        "In physics, quantum mechanics explains the behavior of subatomic particles.",
        "Relativity and gravitation are formulated by Einstein's field equations.",
        "Quantum computing harnesses superposition and entanglement to perform calculations.",
        "Classical mechanics describes the motion of macroscopic objects.",
        "Machine learning models discover statistical patterns from data.",
        "Quantum algorithms like Shor's algorithm can factor integers in polynomial time.",
    ]

    print("=== mini-ir Quickstart Demonstration ===\n")
    print(f"Indexed {len(corpus)} documents.")

    # Initialize IRIndex
    index = IRIndex(corpus, bm25_k1=1.5, bm25_b=0.75)

    query = "quantum algorithms and computing"
    print(f"\nQuery: {query!r}\n")

    # 1. TF-IDF
    print("--- 1. TF-IDF (Cosine Similarity) ---")
    tfidf_results = index.search_tfidf(query, top_k=3)
    for rank, (doc_id, score) in enumerate(tfidf_results, start=1):
        print(f"#{rank} [Score: {score:.4f}] Doc {doc_id}: {corpus[doc_id]}")

    # 2. BM25
    print("\n--- 2. BM25 (Okapi / Robertson-Sparck Jones) ---")
    bm25_results = index.search_bm25(query, top_k=3)
    for rank, (doc_id, score) in enumerate(bm25_results, start=1):
        print(f"#{rank} [Score: {score:.4f}] Doc {doc_id}: {corpus[doc_id]}")

    # 3. Hybrid RRF
    print("\n--- 3. Hybrid Reciprocal Rank Fusion (RRF) ---")
    rrf_results = index.search_hybrid_rrf(query, top_k=3, k=60)
    for rank, (doc_id, score) in enumerate(rrf_results, start=1):
        print(f"#{rank} [Score: {score:.4f}] Doc {doc_id}: {corpus[doc_id]}")


if __name__ == "__main__":
    main()
