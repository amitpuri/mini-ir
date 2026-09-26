"""
Example: Indexing and Searching Custom Documents or Files.
Shows how to index arbitrary text strings, lines from a file, or in-memory notes.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_ir import IRIndex


def main():
    # Example notes or articles
    notes = [
        "Architecture Note: Use event-driven messaging with Kafka for high-throughput stream processing.",
        "Database Guide: PostgreSQL indexing strategies including B-Tree, GIN for full-text search, and GiST.",
        "Security Best Practices: Always sanitize user inputs to prevent SQL injection and cross-site scripting (XSS).",
        "Frontend Tips: Optimize React re-renders using useMemo, useCallback, and React.memo.",
        "DevOps Playbook: Containerize microservices using Docker and orchestrate them with Kubernetes clusters.",
        "Search Systems: Lexical search engines use inverted indices, BM25, and term frequency statistics.",
    ]

    print(f"Indexing {len(notes)} technical notes...\n")
    index = IRIndex(notes)

    queries = [
        "kubernetes containerization devops",
        "sql injection prevention security",
        "search indexing inverted index",
    ]

    for q in queries:
        print(f"Query: {q!r}")
        results = index.search_hybrid_rrf(q, top_k=2)
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"  Rank #{rank} [RRF: {score:.4f}] => {notes[doc_id]}")
        print()


if __name__ == "__main__":
    main()
