"""Extracts the final answer from a model response."""

from __future__ import annotations

import re

# Tried in this order. \boxed{} is the clearest, "Final Answer:" is what the
# prompt asks for, and a plain "Answer:" is what models often write instead.
FINAL_PATTERNS = [
    r"\\boxed\{([^}]+)\}",
    r"(?i)final\s*answer\s*[:=]\s*([^\n]+)",
    r"(?i)answer\s*[:=]\s*([^\n]+)",
]

INTEGER = re.compile(r"(?<!\d)-?\d+(?!\d)")

# characters around the answer that are not part of it, e.g. "$106$"
WRAPPERS = "${}\\ \t"


def normalize(answer):
    s = answer.replace("\\boxed", "").replace("\\text", "")
    s = s.strip().strip(WRAPPERS)
    # second pass, e.g. "$\boxed{106}.$" is only "{106}." at this point
    s = s.strip().strip(".")
    return s.strip().strip(WRAPPERS).strip()


def extract_final_answer(text):
    if not text:
        return None

    for pattern in FINAL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return normalize(match.group(1))

    # no marked answer: fall back to the last integer in the text
    numbers = INTEGER.findall(text)
    return numbers[-1] if numbers else None
