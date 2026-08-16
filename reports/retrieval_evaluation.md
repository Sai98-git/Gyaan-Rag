# MSMARCO-XI Retrieval Evaluation Report

This report documents the design, architecture, and empirical evaluation of the retrieval subsystem for the multilingual `ai4bharat/MSMARCO-XI` dataset.

---

## 1. Retrieval & Indexing Subsystem Architecture

The offline retrieval subsystem is composed of three indexing strategies:
1. **Dense Retrieval**: Utilizes localized transformer-based sentence embeddings and a cosine similarity index.
2. **Sparse Retrieval**: A custom BM25 implementation for lexical keyword matching.
3. **Hybrid Retrieval**: Combines dense and sparse search results using Weighted Reciprocal Rank Fusion (RRF).

The system persists all indexes locally in the `data/indexes/` folder, structured by chunking strategy:
```text
data/
    indexes/
        passage/
            dense/ (embeddings.npy, metadata.json)
            bm25/  (bm25_index.json)
        sliding_window/
            dense/
            bm25/
        semantic/
            dense/
            bm25/
```

---

## 2. Component Design & Settings

### Embedding Model: `intfloat/multilingual-e5-small`
* **Dimension**: 384
* **Supported Languages**: 94 languages (fully supporting Hindi, Bengali, Tamil, Telugu, Urdu, Sanskrit, etc.).
* **Model Size**: ~118M parameters (470 MB file size).
* **Execution**: Runs locally on CPU/GPU.
* **Instruction format**: Prepends `"query: "` to search queries and `"passage: "` to indexed passages to maximize retrieval alignment.

### Custom Dense Index (`NumpyVectorStore`)
* **Similarity**: Cosine similarity. Since embeddings are L2-normalized upon creation, this simplifies to a fast matrix dot-product:
  $$\text{Score}(d, q) = \mathbf{E}_{d} \cdot \mathbf{v}_{q}$$
* **Persistence**: Persisted using Numpy's binary format (`embeddings.npy`) and JSON (`metadata.json`).

### Custom Sparse Index (`BM25Retriever`)
* **Algorithm**: Pure-python word-frequency indexer.
* **Tokenization**: Cleans punctuation and splits words, providing a language-independent lexical baseline.
* **IDF Formula**: Uses a positive-guaranteed IDF formula:
  $$\text{IDF}(q) = \ln\left(\frac{N + 1}{n(q) + 0.5}\right)$$
  This prevents negative score penalization. BM25 parameters are set to $k_1 = 1.5$ and $b = 0.75$.

### Reciprocal Rank Fusion (RRF)
* **Strategy**: Combines Dense and Sparse rank lists up to a candidate depth of 100 docs.
* **Formula**:
  $$\text{RRF\_Score}(d) = w_{\text{dense}} \cdot \frac{1}{60 + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{60 + r_{\text{sparse}}(d)}$$
* **Weights**: Set to equal representation ($w_{\text{dense}} = 0.5, w_{\text{sparse}} = 0.5$) by default.

---

## 3. Evaluation Setup & Relevance Labels

* **Dataset Split**: Validation split (`validation/hinval.parquet` in Hindi).
* **Corpus Size**: `200` records, representing `2,000` original passages.
* **Evaluation Queries**: `104` queries that had at least one selected passage in the index.
* **Relevance Methodology**: Derived from the dataset's built-in `passages.is_selected` labels. A chunk in the index is considered relevant if it has the same `query_id` as the query and its metadata is tagged with `is_selected = 1`.

---

## 4. Benchmark Results & Metric Comparisons

Below is the empirical comparison table generated from the evaluation script over the 104 queries:

| Chunk Strategy | Retrieval Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Index Size (MB) | Mean Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **passage** | Dense | 0.404 | 0.750 | 0.913 | 0.556 | 11.74 MB | 95.03 ms |
| **passage** | BM25 | 0.183 | 0.644 | 0.788 | 0.363 | 11.74 MB | **4.33 ms** |
| **passage** | Hybrid RRF | 0.308 | 0.702 | 0.837 | 0.484 | 11.74 MB | 101.77 ms |
| **sliding_window**| Dense | 0.433 | 0.904 | 0.952 | 0.617 | 9.84 MB | 95.68 ms |
| **sliding_window**| BM25 | 0.308 | 0.798 | 0.865 | 0.488 | 9.84 MB | 3.69 ms |
| **sliding_window**| Hybrid RRF | 0.394 | 0.846 | 0.904 | 0.573 | 9.84 MB | 101.48 ms |
| **semantic** | Dense | 0.413 | **0.923** | **0.971** | **0.625** | **9.20 MB** | 88.81 ms |
| **semantic** | BM25 | 0.288 | 0.817 | 0.856 | 0.521 | 9.20 MB | **2.42 ms** |
| **semantic** | Hybrid RRF | 0.413 | 0.865 | 0.894 | 0.610 | 9.20 MB | 88.23 ms |

---

## 5. RAG Retrieval Performance Analysis

### 1. The Best Chunking Strategy: Semantic/Structure-Aware
**Semantic Chunking** is the clear winner:
* It achieves the highest **Recall@10 (0.971)**, **Recall@5 (0.923)**, and **MRR@10 (0.625)**.
* By grouping highly related adjacent passages, it compresses the segment index size to the smallest size (**9.20 MB**) and achieves the lowest query latency (**88.81 ms**).

### 2. Dense vs. Sparse (BM25)
**Dense Retrieval** outperforms BM25 significantly:
* On Semantic Chunks, Dense gets **0.971 Recall@10** vs BM25's **0.856**.
* Lexical search (BM25) is fast (2.42 ms) but suffers from synonym mismatches in translated Hindi texts, which E5 resolves semantically.

### 3. Why Hybrid (RRF) Underperformed Dense
In RRF rank merging, when the sparse rank list is substantially lower in accuracy (BM25 MRR@10 = 0.521 vs Dense MRR@10 = 0.625), combining them pulls down the higher quality dense ranks. Consequently, pure Dense retrieval yields the highest accuracy in this Indic-language translation setup.

---

## 6. Recommendations for Phase 5 (LLM & Generation)

* **Retriever Config**: Move forward with **Semantic/Structure-Aware Chunking + Dense Retrieval** (using the `intfloat/multilingual-e5-small` model).
* **Storage Optimization**: The persisted Numpy index works extremely well and scales nicely to development size without needing FAISS, and can be easily swapped for a vector database later.
