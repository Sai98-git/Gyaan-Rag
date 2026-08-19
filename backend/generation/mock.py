import time
import json
import logging
from typing import List, Dict, Any
from backend.generation.base import BaseGenerator

logger = logging.getLogger(__name__)

ABSTENTION = "I don't have enough information in the retrieved sources to answer that reliably."

class MockGenerator(BaseGenerator):
    """
    Mock Generator for developer testing and local offline runs.

    Generates a clean, natural-language grounded answer by quoting or
    paraphrasing the most relevant retrieved passage.  It never exposes
    chunk IDs, similarity scores, or internal metadata in the answer text.
    """

    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"Invoking MockGenerator for query: '{query}'")

        # Safe fallback — empty context
        if not context:
            return {
                "answer": ABSTENTION,
                "sources": [],
                "provider": "mock",
            }

        # Simulate local compute latency
        time.sleep(0.15)

        # Build source list for provenance (metadata only, never concatenated into answer)
        sources_list = []
        for chunk in context:
            sources_list.append({
                "chunk_id": chunk["chunk_id"],
                "score": chunk.get("score", 1.0),
                "preview": chunk["text"][:200].strip(),
                "metadata": chunk.get("metadata", {}),
            })

        # Pick the highest-score chunk as the primary grounding basis
        best = max(context, key=lambda c: c.get("score", 0.0))
        best_text = best["text"].strip()

        # If the best passage text is very short, it may not support an answer
        if len(best_text) < 20:
            return {
                "answer": ABSTENTION,
                "sources": sources_list,
                "provider": "mock",
            }

        # Construct a concise, clean natural-language answer from the passage.
        # The answer contains only grounded text — no metadata, IDs, or provider notes.
        # Provider is communicated separately via the 'provider' field.
        truncated = best_text[:600]
        if len(best_text) > 600:
            truncated += "..."

        answer = f"Based on the retrieved passages:\n\n{truncated}"

        return {
            "answer": answer,
            "sources": sources_list,
            "provider": "mock",
        }

    def generate_stream(self, query: str, context: List[Dict[str, Any]]) -> Any:
        res = self.generate(query, context)
        words = res["answer"].split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield json.dumps({"type": "token", "delta": token})
            time.sleep(0.01)
        yield json.dumps({
            "type": "done",
            "answer": res["answer"],
            "sources": res["sources"],
            "provider": "mock"
        })

