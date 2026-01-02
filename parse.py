import re

# Minimal, deterministic extraction. Start simple; refine later.
FINAL_PATTERNS = [
    r"\\boxed\{([^}]+)\}",
    r"(?i)final\s*answer\s*[:=]\s*([^\n]+)",
    r"(?i)answer\s*[:=]\s*([^\n]+)",
]

def normalize(ans: str) -> str:
    return ans.strip().strip(".").strip()

def extract_final_answer(text: str) -> str | None:
    if not text:
        return None

    for pat in FINAL_PATTERNS:
        m = re.search(pat, text)
        if m:
            return normalize(m.group(1))

    # fallback: last integer (useful for AIME-like)
    ints = re.findall(r"(?<!\d)-?\d+(?!\d)", text)
    if ints:
        return ints[-1]

    return None
