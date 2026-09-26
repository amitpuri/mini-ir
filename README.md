<div align="center">

# mini-ir

**A minimal, zero-dependency Information Retrieval engine built from scratch in pure Python.**

[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/amitpuri/mini-ir)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20external-brightgreen.svg)](https://github.com/amitpuri/mini-ir)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-success.svg)](tests/)

*Understand modern search, BM25, TF-IDF vector space scoring, and hybrid Reciprocal Rank Fusion (RRF) — without 2GB of PyTorch, NumPy, or ML bloat.*

</div>

---

## Highlights

- **Zero External Dependencies**: Built entirely using Python's standard library (`math`, `re`, `collections`, `typing`).
- **TF-IDF + Cosine Similarity**: Vector space retrieval with sublinear TF scaling ($1 + \ln(\text{tf})$) and inverse document frequency.
- **BM25 (Okapi / Robertson-Sparck Jones)**: Industry-standard probabilistic lexical ranking with term-frequency saturation ($k_1$) and document-length normalization ($b$).
- **Hybrid Search via RRF**: Reciprocal Rank Fusion ($1 / (k + \text{rank})$) combining sparse TF-IDF and BM25 signals (purely lexical, stdlib-only). Includes a standalone fusion helper to combine external dense vector embeddings.
- **Single-File Drop-in or Package**: Use `mini_ir.py` directly in your project or install via `pip`.
- **Fast & Fully Tested**: 100% test coverage running in under 25ms using Python's built-in `unittest`.

---

## Architecture & Flow

```
                      +-------------------+
                      | Raw Document Text |
                      +---------+---------+
                                |
                 [ Tokenization & Stopwords ]
                                |
                                v
               +---------------------------------+
               |             IRIndex             |
               | - Document Frequencies (df)     |
               | - Term Frequencies (tf)         |
               | - Document Lengths (dl, avgdl)  |
               +----------------+----------------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
+--------------------------+         +--------------------------+
|      TF-IDF Engine       |         |       BM25 Engine        |
| - Sublinear TF Scaling   |         | - TF Saturation (k1)     |
| - Log Inverse Doc Freq   |         | - Length Normalization(b)|
| - Cosine Similarity      |         | - Robertson-SJ IDF       |
+------------+-------------+         +-------------+------------+
             |                                     |
             +------------------+------------------+
                                |
                                v
               +---------------------------------+
               |   Reciprocal Rank Fusion (RRF)  |
               |    score(d) = sum(1 / (k + r))  |
               +----------------+----------------+
                                |
                                v
                    Ranked Hybrid Results
```

---

## How It Works (The Mathematics)

### 1. TF-IDF Vector Space Model
For a query term $t$ and document $d$:

$$\text{TF}(t, d) = 1 + \ln(\text{freq}(t, d)) \quad \text{for } \text{freq} > 0$$

$$\text{IDF}(t) = \ln\left(\frac{N}{\text{df}(t)}\right)$$

Document and query vectors are scored via **Cosine Similarity**:

$$\text{CosineSim}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

---

### 2. BM25 (Robertson-Sparck Jones)
BM25 addresses TF-IDF's lack of term saturation and length normalization:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}_{\text{BM25}}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

where:
- $k_1$ controls term frequency saturation (default `1.5`).
- $b$ controls length normalization penalty (default `0.75`).
- Smoothed IDF guarantees positive scores:

$$\text{IDF}_{\text{BM25}}(t) = \ln\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$

---

### 3. Reciprocal Rank Fusion (RRF)
RRF combines disparate ranking systems without requiring calibrated probabilities or score normalization:

$$\text{Score}_{\text{RRF}}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

where $r_m(d)$ is the rank position (1-indexed) of document $d$ in system $m$, and $k$ is a smoothing constant (standard default `60`).

---

## Quickstart

### Installation

Clone and install locally:
```bash
git clone https://github.com/amitpuri/mini-ir.git
cd mini-ir
pip install -e .
```

*Or simply copy [`mini_ir.py`](mini_ir.py) directly into your project!*

---

### Python API Usage

```python
from mini_ir import IRIndex

corpus = [
    "The cat sat on the mat and looked at the dog.",
    "Dogs are loyal animals often kept as pets.",
    "Python is a popular programming language for data science.",
    "Machine learning models can be trained using Python libraries.",
    "Cats and dogs are the most common household pets.",
    "Data science relies heavily on statistics and programming.",
]

# 1. Build the index
index = IRIndex(corpus, bm25_k1=1.5, bm25_b=0.75)
query = "python programming for machine learning"

# 2. Search using TF-IDF
print("TF-IDF Results:")
for doc_id, score in index.search_tfidf(query, top_k=2):
    print(f"  [{score:.4f}] {corpus[doc_id]}")

# 3. Search using BM25
print("\nBM25 Results:")
for doc_id, score in index.search_bm25(query, top_k=2):
    print(f"  [{score:.4f}] {corpus[doc_id]}")

# 4. Search using Hybrid Reciprocal Rank Fusion (RRF)
print("\nHybrid RRF Results:")
for doc_id, score in index.search_hybrid_rrf(query, top_k=2, k=60):
    print(f"  [{score:.4f}] {corpus[doc_id]}")
```

#### Output:
```text
TF-IDF Results:
  [0.5274] Machine learning models can be trained using Python libraries.
  [0.2421] Python is a popular programming language for data science.

BM25 Results:
  [3.6255] Machine learning models can be trained using Python libraries.
  [2.0846] Python is a popular programming language for data science.

Hybrid RRF Results:
  [0.0328] Machine learning models can be trained using Python libraries.
  [0.0323] Python is a popular programming language for data science.
```

---

## Command Line Interface (CLI)

`mini-ir` can be used directly from your shell:

### Run Interactive Built-in Demo
```bash
python mini_ir.py
```

### Search with Custom Queries & Ranking Methods
```bash
# Evaluate with BM25
python mini_ir.py -q "household pets" -m bm25 -k 2

# Evaluate with Hybrid RRF
python mini_ir.py -q "statistics and data science" -m rrf
```

### Search Any Custom Text File
Index and query your own document corpus (newline-delimited text file):
```bash
python mini_ir.py -f my_notes.txt -q "architecture patterns" -k 5
```

---

## Comparison: TF-IDF vs. BM25 vs. Hybrid RRF

| Feature | TF-IDF | BM25 | Hybrid RRF |
| :--- | :---: | :---: | :---: |
| **Model Type** | Vector Space | Probabilistic Relevance | Rank Aggregation |
| **TF Saturation** | Logarithmic | Non-linear Asymptotic ($k_1$) | Inherited |
| **Length Penalty** | Cosine Norm | Explicit parameter ($b$) | Robust across lengths |
| **Score Scale** | $[0.0, 1.0]$ | $[0.0, +\infty)$ | Rank reciprocal sum |
| **Best For** | Sparse cosine search | General keyword retrieval | Production Hybrid / RAG |

---

## Running Tests

Run the test suite using Python's built-in `unittest` runner:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

No test dependencies needed (`pytest` is also fully compatible).

---

## Scope, Architectural Limits & Trade-offs

`mini-ir` is intentionally designed as an educational, zero-dependency reference implementation. Before using it beyond prototyping, be aware of its deliberate design boundaries:

1. **Purely Lexical (No Dense Vector Retrieval Built-in)**:
   - `mini-ir` implements sparse, lexical algorithms (TF-IDF and BM25) and fuses them using Reciprocal Rank Fusion (RRF).
   - There are **no dense embedding models, neural encoders, or vector DBs bundled** in this library (preserving the zero-dependency standard).
   - However, the standalone function `reciprocal_rank_fusion` can fuse external embedding rank lists (e.g. OpenAI, Cohere, or SentenceTransformers) with BM25. See [`examples/dense_hybrid_fusion.py`](examples/dense_hybrid_fusion.py).

2. **Linear Corpus Scoring ($O(N)$) vs. Inverted Index**:
   - Query evaluation computes scores by iterating over document statistics sequentially.
   - There is no inverted postings list traversal or early termination heuristic (e.g., WAND or Block-Max WAND).
   - For educational exploration, unit testing, and corpora up to tens of thousands of documents, this executes in milliseconds, but it is not intended for multi-million document production scale.

3. **In-Memory & Index Mutation**:
   - Document frequencies and vectors reside in-memory.
   - Adding new documents dynamically requires re-indexing because global corpus statistics ($N$, $\text{avgdl}$, $\text{df}$) change.
   - Persistence is supported via JSON export/import (`IRIndex.save_json` and `IRIndex.load_json`).

4. **Regex Preprocessing & Language Support**:
   - Default tokenization uses a simple regex `[a-z0-9]+` and a hardcoded English stopword list.
   - Non-segmenting languages (CJK: Chinese, Japanese) and RTL scripts are not handled out of the box.
   - There is no stemming (Porter/Snowball) or lemmatization, so "running" and "run" remain separate terms.
   - You can pass a custom tokenizer callable into `IRIndex(documents, tokenizer=custom_fn)` to plug in custom segmenters.

5. **Memory Footprint**:
   - Per-document term counts are stored as standard Python `Counter` and dictionary structures ($O(N \cdot V)$ memory overhead without compression).

---

## Examples

Explore ready-to-run examples in the [`examples/`](examples/) directory:
- [`examples/quickstart.py`](examples/quickstart.py): Basic walkthrough of TF-IDF, BM25, and RRF.
- [`examples/dense_hybrid_fusion.py`](examples/dense_hybrid_fusion.py): Fusing external dense embeddings with BM25 via standalone `reciprocal_rank_fusion`.
- [`examples/search_custom_files.py`](examples/search_custom_files.py): Indexing technical notes and custom knowledge base entries.

---

## Contributing

Contributions, issues, and feature requests are welcome! Please check the [Contributing Guide](CONTRIBUTING.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
