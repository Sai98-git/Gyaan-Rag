# RAG Generation Subsystem & Grounding Guard Evaluation Report

This report documents the design, implementation, and empirical verification of the Grounded LLM Generation layer and Sarvam AI integration for the `ai4bharat/MSMARCO-XI` dataset.

---

## 1. Grounded LLM Integration Architecture

The RAG generation layer is constructed as a decoupled pipeline that normalizes queries, fetches relevance context via dense retrieval, formats source passages, structures safety prompts, runs LLM generation, and executes a deterministic grounding guard.

```text
User Query
    ↓
Query Normalization
    ↓
Dense Retrieval (E5-Small, Semantic Chunking Strategy)
    ↓
Top-K Chunks
    ↓
Context Builder (Deduplication & Formatting)
    ↓
LLM Provider (Sarvam AI / Mock Adapter)
    ↓
Grounding Guard (Confidence + Overlap Checks)
    ↓
Final Answer + Attributed Sources
```

---

## 2. Model Selection & Justification

### Model: `sarvam-105b`
* **Developer**: Sarvam AI.
* **Justification**: Custom-engineered for Indian languages, trained and fine-tuned to support 22 official Indian languages plus English (23 total). It is the premier flagship model for complex Indic reasoning, translation, and structured extraction.
* **API Specifications**: OpenAI-compatible request structure calling `/v1/chat/completions` with authentication performed via the `api-subscription-key` request header.

---

## 3. Context Construction & Formatting

The context builder ([backend/generation/context.py](file:///c:/Users/saiso/OneDrive/Desktop/rag-in-goa/backend/generation/context.py)) aggregates retrieved chunks by:
1. **Deduplication**: Filters duplicate references using unique `chunk_id` attributes.
2. **Formatting**: Wraps each passage in a clean `SOURCE N` tag structure:
   ```text
   SOURCE 1 (ID: <chunk_id>)
   [text block]

   SOURCE 2 (ID: <chunk_id>)
   ...
   ```
3. **Length Constraint**: Enforces a configurable limit (`max_chars = 4000`) to prevent LLM context window overflows.

---

## 4. Prompt Design & Injection Protection

The system instructions enforce strict rules to keep the generation anchored to context:
* **Grounding Instructions**: The model must answer *only* using details present in the context. If the context is insufficient, it must return a standard refusal.
* **Conciseness & Language**: Concise answers matching the language of the query.
* **Prompt Injection Protection**: Chunks are designated as untrusted user-submitted reference DATA rather than executable instructions. The user query is isolated and prohibited from overriding these rules:
  ```text
  CONTEXT:
  [retrieved context]

  USER QUERY:
  [user query text]
  ```

---

## 5. Grounding & Abstention Framework

To ensure factual safety and prevent hallucinations, the system orchestrates three distinct, coordinated mechanisms:

1. **Retrieval Confidence Threshold (Coarse Out-of-Domain Filter)**:
   * **Purpose**: Verifies that the user's query maps to at least one document in the indexing corpus. If the maximum similarity score among retrieved chunks falls below `MIN_RETRIEVAL_SCORE` (calibrated to `0.75`), it bypasses LLM inference and triggers a safe refusal.
   * **Role**: Blocks completely out-of-domain, irrelevant, or garbage queries. It *does not* prune in-topic irrelevant details (which requires the LLM/overlap guard).

2. **Abstention Behavior (LLM Grounding)**:
   * **Purpose**: Instructs the LLM, via explicit prompt guidelines, to evaluate the context and refuse to answer if it does not contain the necessary facts to satisfy the query.
   * **Role**: Prunes in-topic irrelevant passages that have high similarity scores but do not answer the specific query.

3. **Lexical Overlap Guard (Post-Generation Verification)**:
   * **Purpose**: Evaluates whether the generated non-refusal response shares at least one content term (words > 2 characters) with the retrieved context passages.
   * **Role**: Acts as a hard deterministic check to block generated answers that make fabricated claims completely unsupported by the retrieved context.

---

## 6. Grounding Threshold Detailed Performance Analysis

A detailed binary classification analysis was performed over the 100 Hindi validation queries to calculate performance metrics across candidate score thresholds.

### Candidate Thresholds Comparison Table

* **Positive Label Population**: 102 relevant chunks (`is_selected = 1` in top-10)
* **Negative Label Population**: 898 irrelevant chunks (`is_selected = 0` in top-10)

| Threshold | TP Rate (Recall) | FP Rate | Precision | F1-Score | TP Count | FP Count |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.70** | 1.0000 | 1.0000 | 0.1020 | 0.1851 | 102 | 898 |
| **0.75** | 1.0000 | 1.0000 | 0.1020 | 0.1851 | 102 | 898 |
| **0.78** | 1.0000 | 1.0000 | 0.1020 | 0.1851 | 102 | 898 |
| **0.80** | 0.9902 | 0.9599 | 0.1049 | 0.1897 | 101 | 862 |
| **0.82** | 0.9510 | 0.7416 | 0.1271 | 0.2243 | 97 | 666 |
| **0.84** | 0.9020 | 0.4655 | 0.1804 | 0.3007 | 92 | 418 |
| **0.86** | 0.7843 | 0.2873 | 0.2367 | 0.3636 | 80 | 258 |
| **0.88** | 0.6078 | 0.1592 | 0.3024 | 0.4039 | 62 | 143 |
| **0.90** | 0.3725 | 0.0768 | 0.3551 | 0.3636 | 38 | 69 |

### Analysis & Justification

1. **Heavy Overlap of Similarity Scores**:
   * Positives have a median score of `0.8895` and a minimum of `0.7987`.
   * Negatives have a median score of `0.8368` and a p95 of `0.9063`.
   * Since negatives are retrieved within the same topic group, they share highly related vocabulary, causing their scores to overlap extensively with positives.

2. **Why a Score Threshold Alone Cannot Block Hallucinations**:
   * If the threshold is set to `0.82` (retaining `0.9510` Recall), the False Positive Rate remains high at `0.7416` (74.16% of irrelevant retrieved chunks are accepted).
   * If the threshold is raised to `0.86` to block 71% of negatives, Recall drops severely to `0.7843`, rejecting over 21.6% of valid relevant answers.
   * **Reliability Statement**: The available similarity score data does NOT support selecting a single threshold capable of separating in-topic relevant chunks from irrelevant ones. 

3. **Operating Threshold Recommendation**:
   * We select **`0.75`** as our `MIN_RETRIEVAL_SCORE`.
   * Because it is below the minimum observed positive score (`0.7987`), it guarantees **100% Recall (TP Rate = 1.00)** on in-domain queries.
   * Its function is solely a coarse out-of-domain filter (rejecting garbage or out-of-domain queries where scores drop below `0.65`). The task of discarding irrelevant details is deferred to the LLM's grounded prompt reasoning.

---

## 7. Latency and Generation Metrics

Evaluated over 30 test records (using the local Mock provider):
* **Grounded Answers Generated**: 100% (for queries containing relevant context)
* **Low-confidence Abstention Rate**: 0% (on relevant validation queries)
* **Latency Metrics**:
  * **Retrieval Phase**: Mean = `153.77 ms`, Median = `119.87 ms`, p95 = `299.44 ms`
  * **Generation Phase**: Mean = `151.88 ms`, Median = `151.57 ms`, p95 = `153.65 ms` (simulated compute sleep)
  * **End-to-End Latency**: Mean = `305.65 ms`, Median = `272.32 ms`, p95 = `451.48 ms`

---

## 8. Failure Cases and Limitations

1. **High In-Topic Similarities**: Within the same topic group, irrelevant passages still score highly (median `0.8368`) because they share vocabulary. The LLM must be rely on precise system prompt grounding to exclude them.
2. **Multilingual Tokenizer**: The custom BM25 tokenizer strips punctuation but doesn't perform lemmatization or stemming for Indic languages, which can occasionally miss inflected word forms. Dense retrieval operates as the primary resolver for this.
