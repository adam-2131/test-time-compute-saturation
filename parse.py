
from __future__ import annotations

import re


FINAL_PATTERNS = [
    r"\\boxed\{([^}]+)\}",
    r"(?i)final\s*answer\s*[:=]\s*([^\n]+)",
    r"(?i)answer\s*[:=]\s*([^\n]+)",
]

INTEGER = re.compile(r"(?<!\d)-?\d+(?!\d)")

WRAPPERS = "${}\\ \t"


def normalize(answer):
    s = answer.replace("\\boxed", "").replace("\\text", "")
    s = s.strip().strip(WRAPPERS)
    s = s.strip().strip(".")
    return s.strip().strip(WRAPPERS).strip()


def extract_final_answer(text):
    if not text:
        return None

    for pattern in FINAL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return normalize(match.group(1))

    numbers = INTEGER.findall(text)
    return numbers[-1] if numbers else None
