"""
Central orchestration layer for the Medical Multimodal AI Assistant.

Responsibilities:
- Normalize multimodal inputs into text
- Decide when to use RAG
- Enforce medical safety rules
- Call LLM for final response
- Return structured, auditable output
"""

from typing import Optional, Dict
import time

from app.rag.retriever import retrieve_medical_context
from app.safety.guardrails import pre_check_safety, post_check_safety
from app.utils.logger import log_event
from app.core.llm import generate_answer


def handle_user_request(
    user_text: Optional[str] = None,
    image_summary: Optional[str] = None,
) -> Dict:
    """
    Orchestrates the full request flow.

    Args:
        user_text (str): Text from user (typed or transcribed from voice)
        image_summary (str): Textual summary extracted from image (optional)

    Returns:
        dict with:
            - answer (str)
            - sources (list)
            - safety_notes (list)
            - latency_ms (int)
    """

    start_time = time.time()

    # -----------------------------
    # Step 1: Input validation
    # -----------------------------
    if not user_text and not image_summary:
        return {
            "answer": "I did not receive any input to process.",
            "sources": [],
            "safety_notes": ["no_input"],
            "latency_ms": 0,
        }

    # -----------------------------
    # Step 2: Build enriched query
    # -----------------------------
    enriched_query = user_text or ""

    if image_summary:
        enriched_query += (
            "\n\nImage findings (for context only):\n"
            f"{image_summary}"
        )

    # -----------------------------
    # Step 3: Safety pre-check
    # -----------------------------
    safety_flags = pre_check_safety(enriched_query)

    if safety_flags.get("block"):
        return {
            "answer": safety_flags["message"],
            "sources": [],
            "safety_notes": safety_flags["reasons"],
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    # -----------------------------
    # Step 4: Decide if RAG is needed
    # -----------------------------
    # Simple rule for now: medical questions → RAG
    use_rag = True

    rag_context = ""
    sources = []

    if use_rag:
        retrieval_result = retrieve_medical_context(enriched_query)

        rag_context = retrieval_result["context"]
        sources = retrieval_result["sources"]

    # -----------------------------
    # Step 5: LLM generation
    # -----------------------------
    answer = generate_answer(
        user_query=user_text,
        context=rag_context,
    )

    # -----------------------------
    # Step 6: Safety post-check
    # -----------------------------
    final_answer, post_flags = post_check_safety(answer)

    # -----------------------------
    # Step 7: Logging
    # -----------------------------
    log_event(
        event_type="orchestration",
        payload={
            "used_rag": use_rag,
            "num_sources": len(sources),
            "safety_flags": post_flags,
        }
    )

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "answer": final_answer,
        "sources": sources,
        "safety_notes": post_flags,
        "latency_ms": latency_ms,
    }
