"""
Medical safety guardrails.

These functions implement deterministic, explainable rules
to prevent unsafe medical advice.
"""

import re
from typing import Dict, List, Tuple


# -----------------------------
# Rule definitions
# -----------------------------

HARD_BLOCK_PATTERNS = [
    r"\bdiagnose\b",
    r"\bdiagnosis\b",
    r"\bprescribe\b",
    r"\bdosage\b",
    r"\bhow much (mg|ml)\b",
    r"\bshould i take\b",
    r"\bemergency\b",
    r"\bsuicide\b",
    r"\bkill myself\b",
]

SOFT_WARNING_PATTERNS = [
    r"\btreatment\b",
    r"\bcure\b",
    r"\bbest medicine\b",
    r"\bmedication\b",
]

DISCLAIMER_TEXT = (
    "This information is for educational purposes only and "
    "is not a medical diagnosis. Please consult a qualified "
    "healthcare professional for personalized advice."
)


# -----------------------------
# Pre-check guardrail
# -----------------------------
def pre_check_safety(text: str) -> Dict:
    """
    Checks user input for unsafe medical intent BEFORE LLM call.
    """

    lowered = text.lower()
    reasons: List[str] = []

    # Hard blocks
    for pattern in HARD_BLOCK_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append(f"hard_block:{pattern}")

    if reasons:
        return {
            "block": True,
            "message": (
                "I can't help with diagnosis, prescriptions, or emergency medical decisions. "
                "Please consult a qualified healthcare professional."
            ),
            "reasons": reasons,
        }

    # Soft warnings (allowed but flagged)
    for pattern in SOFT_WARNING_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append(f"soft_warning:{pattern}")

    return {
        "block": False,
        "message": None,
        "reasons": reasons,
    }


# -----------------------------
# Post-check guardrail
# -----------------------------
def post_check_safety(answer: str) -> Tuple[str, List[str]]:
    """
    Sanitizes LLM output AFTER generation.
    """

    flags: List[str] = []
    sanitized_answer = answer

    # Overconfidence patterns
    if re.search(r"\byou should\b", answer.lower()):
        flags.append("overconfident_language")

    if re.search(r"\btake \d+ ?(mg|ml)\b", answer.lower()):
        flags.append("dosage_detected")

    # If dangerous patterns detected, override response
    if "dosage_detected" in flags:
        return (
            "I can’t provide specific medication dosages. "
            "Please consult a licensed medical professional.",
            flags,
        )

    # Always append disclaimer for medical answers
    if DISCLAIMER_TEXT.lower() not in answer.lower():
        sanitized_answer = f"{answer}\n\n{DISCLAIMER_TEXT}"
        flags.append("disclaimer_added")

    return sanitized_answer, flags
