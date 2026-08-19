import re
import json
import logging
import requests
from typing import List, Dict, Any, Generator
from backend.core.config import settings
from backend.generation.base import BaseGenerator
from backend.generation.context import format_context
from backend.voice.retry import execute_with_retry
from backend.generation.guard import is_devanagari, get_localized_abstention

logger = logging.getLogger(__name__)

# Reusable persistent HTTP session for TCP keep-alive and connection pooling
_session = requests.Session()

SYSTEM_PROMPT_HI = """आप एक भारतीय ज्ञानकोष के लिए सटीक प्रश्न-उत्तर प्रणाली हैं।
नियम:
1. उत्तर केवल नीचे दिए गए CONTEXT से दें।
2. 1-2 सीधे, स्पष्ट वाक्यों में उत्तर दें। कोई आंतरिक विचार (thoughts), विश्लेषण (analysis) या तर्क (reasoning) न लिखें।
3. यदि संदर्भ में उत्तर नहीं है, तो केवल यह लिखें: मुझे उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"""

SYSTEM_PROMPT_EN = """You are a grounded question-answering assistant.
STRICT RULES:
1. Answer the question directly in 1-2 concise sentences using ONLY facts in the CONTEXT.
2. Do NOT output any internal thoughts, reasoning steps, or analysis.
3. If the context does not contain sufficient facts to answer, respond ONLY with:
"I don't have enough information in the retrieved sources to answer that reliably." """


def sanitize_llm_answer(raw_text: str, is_hindi: bool = False) -> str:
    """
    Strips any chain-of-thought, reasoning steps, preambles, or analysis
    that an LLM might emit. Returns only the direct factual answer.
    """
    text = raw_text.strip()
    
    # 1. If text contains synthesized answer demarcations, extract the final answer
    for marker in [r'\*\*Synthesize[^\*]*\*\*', r'Draft:', r'Final Answer:', r'Answer:']:
        parts = re.split(marker, text, flags=re.IGNORECASE)
        if len(parts) > 1 and len(parts[-1].strip()) > 10:
            text = parts[-1].strip()
            break
    
    # 2. Strip numbered thought steps like '1. Analyze...' if present
    if re.match(r'^(?:\d+\.\s+[\*\w\s\(\)]+:\s*)+', text) or text.startswith("1. "):
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content_lines = []
        for line in lines:
            if re.match(r'^(?:\d+\.\s+|\*\s+)(?:\*\*)?(?:Analyze|Scan|Review|Apply|Evaluate|Look|Synthesize)', line, re.IGNORECASE):
                continue
            if line.startswith("*") and any(k in line for k in ["Mentions", "Describes", "not relevant", "highly relevant"]):
                continue
            content_lines.append(line)
        if content_lines:
            text = "\n".join(content_lines).strip()
            
    # 3. Remove leading/trailing quote marks
    text = re.sub(r'^[\"\']|[\"\']$', '', text).strip()
    
    if not text or len(text) < 5:
        return "मुझे उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।" if is_hindi else "I don't have enough information in the retrieved sources to answer that reliably."
        
    return text


class SarvamGenerator(BaseGenerator):
    """
    Adapter for Sarvam AI Chat Completions API with persistent connection pooling
    and low-latency Server-Sent Events (SSE) token streaming.
    """
    
    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        model_name = settings.SARVAM_MODEL or "sarvam-105b-conversations"
        if model_name == "sarvam-105b":
            model_name = "sarvam-105b-conversations"
        self.model = model_name
        self.base_url = "https://api.sarvam.ai/v1/chat/completions"
        self.timeout = 25.0
        
        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            logger.warning("Sarvam API key is missing or not configured in environment variables.")

    def _prepare_payload(self, query: str, context: List[Dict[str, Any]], stream: bool = False) -> tuple:
        is_hi = is_devanagari(query)
        system_prompt = SYSTEM_PROMPT_HI if is_hi else SYSTEM_PROMPT_EN
        
        formatted_context = format_context(context)
        user_prompt = f"CONTEXT:\n{formatted_context}\n\nUSER QUERY:\n{query}"
        
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 200,
            "stream": stream
        }
        return headers, payload, is_hi

    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            raise ValueError("Sarvam API key is missing. Please set SARVAM_API_KEY in your environment.")
            
        headers, payload, is_hi = self._prepare_payload(query, context, stream=False)
        
        def _do_request() -> requests.Response:
            res = _session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            res.raise_for_status()
            return res
        
        try:
            response = execute_with_retry(
                _do_request,
                max_retries=2,
                initial_delay=0.3,
                backoff_factor=1.5,
                operation_name="Sarvam Chat Completion"
            )
            res_data = response.json()
            choices = res_data.get("choices", [])
            if not choices:
                answer = get_localized_abstention(query)
            else:
                msg = choices[0].get("message", {})
                raw_content = msg.get("content") or msg.get("reasoning_content") or ""
                answer = sanitize_llm_answer(str(raw_content), is_hindi=is_hi)
            
            sources = []
            for chunk in context:
                sources.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "score": chunk.get("score", 0.0),
                    "preview": str(chunk.get("text") or "")[:200].strip(),
                    "metadata": chunk.get("metadata", {})
                })
                
            return {
                "answer": answer,
                "sources": sources,
                "provider": "sarvam"
            }
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Sarvam API timeout: {e}")
            raise TimeoutError("The generation request to Sarvam AI API timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Sarvam API error: {e}")
            raise RuntimeError(f"Sarvam API failure: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            raise RuntimeError(f"Unexpected generation failure: {e}")

    def generate_stream(self, query: str, context: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        Streams generated answer tokens progressively using SSE chunks.
        Yields JSON strings formatted as Server-Sent Events.
        """
        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            abstain_msg = get_localized_abstention(query)
            yield json.dumps({"type": "token", "delta": abstain_msg})
            yield json.dumps({"type": "done", "answer": abstain_msg, "sources": []})
            return

        headers, payload, is_hi = self._prepare_payload(query, context, stream=True)
        
        sources = []
        for chunk in context:
            sources.append({
                "chunk_id": chunk.get("chunk_id"),
                "score": chunk.get("score", 0.0),
                "preview": str(chunk.get("text") or "")[:200].strip(),
                "metadata": chunk.get("metadata", {})
            })

        accumulated_chunks = []
        
        try:
            res = _session.post(
                self.base_url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.timeout
            )
            res.raise_for_status()
            
            for line in res.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                            if delta:
                                accumulated_chunks.append(delta)
                                yield json.dumps({"type": "token", "delta": delta})
                    except Exception:
                        pass
                        
            full_raw = "".join(accumulated_chunks)
            sanitized = sanitize_llm_answer(full_raw, is_hindi=is_hi)
            yield json.dumps({
                "type": "done",
                "answer": sanitized,
                "sources": sources,
                "provider": "sarvam"
            })
            
        except Exception as e:
            logger.error(f"Streaming error from Sarvam: {e}")
            fallback_ans = get_localized_abstention(query)
            yield json.dumps({"type": "token", "delta": fallback_ans})
            yield json.dumps({
                "type": "done",
                "answer": fallback_ans,
                "sources": sources,
                "provider": "sarvam"
            })
