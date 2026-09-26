"""
Example: Fusing External Dense Embeddings with BM25 via Reciprocal Rank Fusion.

This script demonstrates how to build a true dense + sparse hybrid search
pipeline by combining external vector scores (e.g., from an embedding model)
with mini-ir's BM25 lexical ranking using the standalone `reciprocal_rank_fusion` function.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_ir import IRIndex, reciprocal_rank_fusion


def main():
    corpus = [
        "How to train neural networks with PyTorch and autograd.",
        "A guide to baking sourdough bread and maintaining a starter.",
        "Optimizing SQL database performance with compound indexes.",
        "Deep learning architectures for natural language processing.",
        "French pastry baking techniques and croissant lamination.",
        "Distributed database replication and consensus protocols like Raft.",
    ]

    # 1. Lexical search using mini-ir BM25
    index = IRIndex(corpus)
    query = "deep learning machine learning"
    bm25_ranked = index.search_bm25(query, top_k=len(corpus))

    print(f"Query: {query!r}\n")
    print("--- BM25 Lexical Ranking ---")
    for doc_id, score in bm25_ranked[:3]:
        print(f"  [{score:.4f}] Doc {doc_id}: {corpus[doc_id]}")

    # 2. Simulated dense semantic search results (e.g. from cosine similarity of embeddings)
    # Suppose a dense embedding model captures semantic similarity without exact keyword match:
    # "How to train neural networks" (Doc 0) is semantically ranked #1 by the vector model
    dense_ranked = [
        (0, 0.92),  # Neural networks training
        (3, 0.88),  # Deep learning architectures
        (5, 0.45),  # Distributed systems
        (2, 0.40),  # SQL databases
        (4, 0.12),  # Pastry
        (1, 0.10),  # Sourdough
    ]

    print("\n--- Dense Vector Semantic Ranking (Simulated Embeddings) ---")
    for doc_id, score in dense_ranked[:3]:
        print(f"  [{score:.4f}] Doc {doc_id}: {corpus[doc_id]}")

    # 3. Fuse Dense + Sparse using Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60, top_k=3)

    print("\n--- Fused Hybrid (Dense + BM25 via RRF) Ranking ---")
    for doc_id, score in fused_results:
        print(f"  [{score:.4f}] Doc {doc_id}: {corpus[doc_id]}")


if __name__ == "__main__":
    main()
