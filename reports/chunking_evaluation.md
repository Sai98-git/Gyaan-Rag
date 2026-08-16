# MSMARCO-XI Chunking Evaluation Report

This report evaluates and compares multiple chunking strategies implemented for processing the multilingual `ai4bharat/MSMARCO-XI` dataset.

---

## 1. Rationale: Why Multiple Strategies Are Needed

In Retrieval-Augmented Generation (RAG) pipelines, the way text is partitioned directly determines retrieval precision and context relevance:
* **Naive Chunking** (splitting at arbitrary character or token counts) often splits words or sentence boundaries in half, destroying coherence and degrading embedding quality.
* **Passage-Aware Chunking** leverages the natural structural boundaries created by the translators/curators of the dataset.
* **Sliding-Window Chunking** ensures overlap to prevent information loss at chunk boundaries, which is crucial for facts that span multiple sentences.
* **Semantic/Structure-Aware Chunking** dynamically groups highly related adjacent sections while splitting oversized blocks, balancing context density and granularity.

Evaluating these strategies on Indic-language texts allows us to determine the optimal trade-off between semantic integrity, size distribution, and compute overhead.

---

## 2. MSMARCO-XI Record Structure

An inspection of the normalized `ai4bharat/MSMARCO-XI` schema shows that each record has the following passage structure under the `passages` key:

* **Passages count per record**: Typically **10 passages** per query/row.
* **Fields available in passages**:
  * `English_passages` (List of strings): Original English search snippets.
  * `Translated_passages` (List of strings): Translated snippets (e.g. Hindi).
  * `is_selected` (List of integers): Binary flags (`0` or `1`) indicating whether the passage was marked as containing the correct answer.
* **Title / URL availability**: **No titles or URLs exist** within the passages or the repository.
* **Language metadata**: Passages do not have independent language attributes, but the parent record preserves `source_lang` ("en") and `target_lang` (e.g. "hin_Deva" / "hi").
* **Ordering significance**: **Yes, the ordering is highly meaningful**. The order represents search relevance ranking in the original MS MARCO retrieval stage.

---

## 3. Implementations of Chunking Strategies

### Strategy A: Passage-Aware Chunking
* **Description**: Splits text cleanly along the original 10 passage boundaries.
* **Preservation**: Completely preserves the natural paragraph boundaries. Each chunk contains exactly one original passage.
* **ID Format**: `{query_id}_passage_{index}`.

### Strategy B: Sliding-Window Chunking
* **Description**: Concatenates all passages and slides a window based on sentence boundaries.
* **Indic sentence splitter**: Utilizes a regex matching standard Indic sentence delimiters (`।` / purna viram, `॥`, `|`) as well as English punctuation (`.`, `?`, `!`).
* **Configurations**: Respects `CHUNK_SIZE` (500 chars), `CHUNK_OVERLAP` (50 chars), and `MIN_CHUNK_SIZE` (50 chars). It rolls backwards at sentence offsets to maintain exact character overlap limits without breaking words.
* **ID Format**: `{query_id}_sliding_{index}`.

### Strategy C: Semantic/Structure-Aware Chunking
* **Description**: A lightweight, embedding-free semantic chunker.
* **Algorithm**:
  1. Any passage exceeding `MAX_CHUNK_SIZE` (1,000 chars) is split into sentence groups.
  2. Adjacent passages are tokenized and compared using **Jaccard Word Similarity**:
     $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  3. If Jaccard similarity is $\ge 0.08$ (sharing context/entities) and the combined length is $\le \text{MAX\_CHUNK\_SIZE}$, they are merged into a single semantic chunk. Otherwise, a new chunk is started.
* **ID Format**: `{query_id}_semantic_{index}`.

---

## 4. Benchmark Results & Statistics

The strategies were evaluated on a representative sample of **100 records** (comprising 1,000 original passages) from `validation/hinval.parquet`:

| Metric | Passage-Aware | Sliding-Window | Semantic/Structure-Aware |
| :--- | :--- | :--- | :--- |
| **Total Chunks Generated** | 997 | 767 | 482 |
| **Average Chunk Length (chars)** | 316.5 | 418.6 | 656.8 |
| **Median Chunk Length (chars)** | 293.0 | 435.0 | 716.5 |
| **Minimum Chunk Length (chars)** | 53 | 52 | 80 |
| **Maximum Chunk Length (chars)** | 7515 | 7621 | 7515 |
| **Processing Time (s)** | 0.0105s | 0.0476s | 0.3437s |

### Data Quality Findings (Maximum Size Anomaly)
The maximum chunk size for all strategies reaches **~7.5k characters**. This indicates that the dataset contains a few anomalous records where a single sentence (or a text block entirely lacking punctuation/purna virams) spans 7.5k characters, preventing sentence-boundary chunkers from splitting it.

---

## 5. Metadata Schema Preservation

Every chunk contains a nested `metadata` dictionary to ensure 100% traceability to the original dataset record:

* **Passage-Aware**:
  `source_passage_id`, `original_passage_position`, `language`, `dataset_split`, `dataset_language`, `parent_query_id`, `is_selected`, `english_passage`
* **Sliding-Window**:
  `strategy`, `position`, `language`, `dataset_split`, `dataset_language`, `parent_query_id`, `start_sentence_idx`, `end_sentence_idx`
* **Semantic/Structure-Aware**:
  `original_positions`, `language`, `dataset_split`, `dataset_language`, `parent_query_id`, `is_selected`, `english_passages` (lists of all merged components)

---

## 6. Strengths and Weaknesses of Each Strategy

### Passage-Aware Chunking
* **Strengths**: Extremely fast (0.01s for 100 records); perfectly preserves the retrieval boundaries curated by human editors; includes precise selected/unselected labels.
* **Weaknesses**: Can yield very short chunks (e.g. 53 chars) leading to fragmented context, or oversized chunks (up to 7.5k chars) if the original passage was poorly formatted.

### Sliding-Window Chunking
* **Strengths**: Configurable overlap guarantees no information loss between chunk boundaries; enforces size limits while respecting sentence boundaries.
* **Weaknesses**: Slices across natural passage boundaries, potentially blending unrelated search passages together in the same window.

### Semantic/Structure-Aware Chunking
* **Strengths**: Minimizes fragment counts (reduces chunks from 997 to 482) by merging related consecutive passages; lightweight and does not require costly GPU/API embeddings.
* **Weaknesses**: Merging depends on word overlap heuristic (Jaccard similarity), which might occasionally group items that are superficially similar but discuss different aspects.

---

## 7. Recommendations for the Retrieval Phase

For the upcoming retrieval evaluation, we recommend carrying forward **both the Passage-Aware and the Semantic/Structure-Aware strategies**:
1. **Passage-Aware** represents the baseline "gold-standard" boundaries. It is highly aligned with how the original MS MARCO answers were extracted.
2. **Semantic/Structure-Aware** provides dense, grouped context chunks which will likely improve dense retrieval (vector embeddings) by reducing the total index size and increasing context window density during LLM generation.

We will benchmark retrieval quality (NDCG@K, Recall) for both strategies once the embedding and retrieval layers are implemented.
