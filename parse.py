"""Conservative answer extraction and exact normalization, used both by the
experiment (eval.py) and by the post-hoc rescoring (scripts/replay_corrected.py).

Every rule below is applied to every response in the same way; none refers to
a particular problem, model or dataset.

Extraction
  1. The last explicit "Final Answer:" label in the response wins. Markdown
     (bold, headings, bullets, backticks) and a LaTeX \\text{...} wrapper around
     the label are ignored when looking for it. If nothing follows the label on
     its line, the answer is read from what comes next: a display-math block,
     a \\boxed{...} (which may span lines), or the next non-empty line.
  2. Only if the response contains no label at all, the last complete,
     brace-balanced \\boxed{...} is used.
  3. Otherwise the response has no valid answer (None). There is no
     last-integer fallback.
  4. A token-limit response (completion_tokens reached max_tokens) keeps an
     answer only through rule 1, and only when that answer is complete: the
     label line, or the block that follows it, is closed and followed by a
     line break. Rule 2 is never applied to such responses.

Normalization (exact; changes representation only, never the value)
  - Numbers: thousands separators, leading zeros, trailing decimal zeros, a
    leading "+", "-0" and the Unicode minus are canonicalized. No rounding.
  - Percentages keep the percent sign, so "50%" and "50" stay different.
  - Currency signs, degree signs and unit words stay part of the answer.
  - \\frac{a}{b} and a/b are written as "a/b"; fractions are not reduced and
    not converted to decimals.
  - "<symbolic name> = <value>" (e.g. "m + n = 106") becomes "<value>".
  - Sentences, hedged answers and multiple answers ("or", "and", "\\approx",
    "approximately", "\\pm") are not reliable answers and become None.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

LABEL_RE = re.compile(r"final\s+answer\s*[:=]", re.IGNORECASE)
TEXT_LABEL_RE = re.compile(
    r"\\(?:text|textbf|mathrm|mathbf)\s*\{\s*(final\s+answer\s*[:=][^{}]*)\}",
    re.IGNORECASE,
)
INNER_TEXT_RE = re.compile(r"\\(?:text|textbf|textrm|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
BOXED_RE = re.compile(r"\\boxed\s*\{")

NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+"
NUM_RE = re.compile(rf"(?:{NUM})")
PERCENT_RE = re.compile(rf"({NUM})\s*(?:%|percent)", re.IGNORECASE)
CURRENCY_RE = re.compile(rf"\$\s*({NUM})")
DEGREE_RE = re.compile(
    rf"({NUM})\s*(?:\^\s*\{{?\s*\\circ\s*\}}?|°|\\degree|degrees?)", re.IGNORECASE
)
FRAC_RE = re.compile(rf"(-?)\s*\\[dt]?frac\s*\{{\s*({NUM})\s*\}}\s*\{{\s*({NUM})\s*\}}")
SLASH_RE = re.compile(rf"({NUM})\s*/\s*({NUM})")
UNIT_RE = re.compile(rf"({NUM})\s+([A-Za-z]+(?:\s+[A-Za-z]+){{0,2}})(\^\{{?\d\}}?)?")
LHS_RE = re.compile(r"[A-Za-z\\][A-Za-z0-9_^{}\\+\-*()\s]{0,24}")
HEDGE_RE = re.compile(r"\\approx|≈|\\pm|±|\bapprox(?:imately)?\b|\bor\b|\band\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")
SEPARATORS = {"---", "***", "___"}


def _is_escaped(text: str, index: int) -> bool:
    count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        count += 1
        index -= 1
    return count % 2 == 1


def _braced(text: str, open_index: int) -> tuple[str, int] | None:
    """Content of the balanced {...} starting at open_index, and the index after it."""
    depth = 0
    for i in range(open_index, len(text)):
        ch = text[i]
        if ch == "{" and not _is_escaped(text, i):
            depth += 1
        elif ch == "}" and not _is_escaped(text, i):
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i], i + 1
    return None


def _boxed_spans(text: str) -> list[tuple[str, int]]:
    """(content, end_index) of every complete \\boxed{...} in text."""
    spans = []
    for m in BOXED_RE.finditer(text):
        parsed = _braced(text, m.end() - 1)
        if parsed is not None:
            spans.append(parsed)
    return spans


def _strip_markdown(line: str) -> str:
    s = line.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"^\s*(?:#{1,6}|>|[-*•]\s)\s*", "", s)
    return s.strip()


def _balanced(s: str) -> bool:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{" and not _is_escaped(s, i):
            depth += 1
        elif ch == "}" and not _is_escaped(s, i):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _open_braces(s: str) -> int:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{" and not _is_escaped(s, i):
            depth += 1
        elif ch == "}" and not _is_escaped(s, i) and depth:
            depth -= 1
    return depth


def _clean(s: str) -> str:
    """Remove formatting around an answer; never changes its content."""
    for _ in range(12):
        before = s
        s = _strip_markdown(s)
        s = re.sub(r"^\*(.+)\*$", r"\1", s)
        s = re.sub(r"^final\s+answer\s*[:=]\s*", "", s, flags=re.IGNORECASE)
        s = s.strip().rstrip(".").strip()

        for opener, closer in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$")):
            if s.startswith(opener) and s.endswith(closer) and len(s) >= len(opener) + len(closer):
                s = s[len(opener):len(s) - len(closer)].strip()
            elif s.endswith(closer) and opener not in s:
                s = s[:len(s) - len(closer)].strip()
            elif s.startswith(opener) and closer not in s[len(opener):]:
                s = s[len(opener):].strip()
        if len(s) >= 2 and s.startswith("$") and s.endswith("$") and "$" not in s[1:-1]:
            s = s[1:-1].strip()

        for command in (r"\boxed", r"\text", r"\textbf", r"\mathrm", r"\mathbf"):
            m = re.match(re.escape(command) + r"\s*\{", s)
            if m:
                parsed = _braced(s, m.end() - 1)
                if parsed is not None:
                    inner, end = parsed
                    rest = s[end:].strip()
                    if command != r"\boxed" or not rest:
                        s = (inner + " " + rest).strip()
                break

        s = INNER_TEXT_RE.sub(r"\1", s)
        s = re.sub(r"\\displaystyle|\\left|\\right", "", s)
        s = re.sub(r"\\[,;:! ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s == before:
            break
    return s


def _number(token: str) -> str | None:
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _canonical(s: str) -> str | None:
    s = s.replace("−", "-").replace(r"\%", "%").replace(r"\$", "$").strip()
    if not s or HEDGE_RE.search(s):
        return None

    if s.count("=") == 1:
        lhs, rhs = (part.strip() for part in s.split("="))
        if rhs and LHS_RE.fullmatch(lhs) and re.search(r"[A-Za-z]", lhs):
            s = _clean(rhs)
        else:
            return None
    elif "=" in s:
        return None

    if NUM_RE.fullmatch(s):
        return _number(s)
    if m := PERCENT_RE.fullmatch(s):
        return f"{_number(m.group(1))}%"
    if m := CURRENCY_RE.fullmatch(s):
        return f"${_number(m.group(1))}"
    if m := DEGREE_RE.fullmatch(s):
        return f"{_number(m.group(1))}°"
    if m := FRAC_RE.fullmatch(s):
        return f"{m.group(1)}{_number(m.group(2))}/{_number(m.group(3))}"
    if m := SLASH_RE.fullmatch(s):
        return f"{_number(m.group(1))}/{_number(m.group(2))}"
    if m := UNIT_RE.fullmatch(s):
        return f"{_number(m.group(1))} {m.group(2).lower()}{m.group(3) or ''}"

    words_only = re.sub(r"\\[A-Za-z]+", " ", s)
    if (
        len(s) > 60
        or s.endswith(":")
        or not _balanced(s)
        or SENTENCE_RE.search(words_only)
        or not re.search(r"[A-Za-z0-9]", s)
    ):
        return None
    return s


def normalize_answer(raw: str | None) -> str | None:
    """Canonical form of one answer string, or None if it is not a reliable answer."""
    if raw is None:
        return None
    return _canonical(_clean(str(raw)))


def normalize_gold(answer: str) -> set[str]:
    """Accepted canonical reference answers.

    The AIME source dataset marks a problem with two accepted answers as
    "<a> or <b> (both were accepted)"; that is the only multi-answer form.
    """
    m = re.fullmatch(r"\s*(\S+)\s+or\s+(\S+)\s*\(both were accepted\)\s*", str(answer), re.IGNORECASE)
    parts = [m.group(1), m.group(2)] if m else [str(answer)]
    return {a for a in (normalize_answer(p) for p in parts) if a is not None}


def _after_empty_label(lines: list[str], start: int) -> tuple[str | None, int]:
    """Answer text that follows a label with nothing after it, and the last line it used."""
    i = start
    while i < len(lines) and (not lines[i].strip() or _strip_markdown(lines[i]) in SEPARATORS):
        i += 1
    if i >= len(lines):
        return None, len(lines) - 1

    first = lines[i].strip()
    for opener, closer in ((r"\[", r"\]"), ("$$", "$$")):
        if first.startswith(opener):
            body = first[len(opener):]
            j = i
            while closer not in body:
                j += 1
                if j >= len(lines):
                    return None, len(lines) - 1
                body += "\n" + lines[j]
            return body[:body.index(closer)], j

    if BOXED_RE.search(first):
        rest = "\n".join(lines[i:])
        m = BOXED_RE.search(rest)
        parsed = _braced(rest, m.end() - 1)
        if parsed is None:
            return None, len(lines) - 1
        inner, end = parsed
        return inner, i + rest[:end].count("\n")

    return first, i


def extract_answer(text: str | None, *, token_limit: bool = False) -> str | None:
    """Canonical final answer of one model response, or None."""
    if not text:
        return None
    lines = text.splitlines()

    label = None
    for i, raw in enumerate(lines):
        line = TEXT_LABEL_RE.sub(r"\1", _strip_markdown(raw))
        matches = list(LABEL_RE.finditer(line))
        if matches:
            content = line[matches[-1].end():].strip()
            # A wrapper opened before the label, as in \boxed{Final Answer: 955},
            # leaves its closing braces after the answer.
            opened = _open_braces(line[:matches[-1].start()])
            while opened and content.endswith("}") and not _balanced(content):
                content = content[:-1].rstrip()
                opened -= 1
            label = (i, content)

    if label is not None:
        i, content = label
        last_line = i
        if not content or content in {r"\[", "$$"}:
            offset = i if content else i + 1
            tail = ([content] if content else []) + lines[i + 1:]
            content, used = _after_empty_label(tail, 0)
            if content is None:
                return None
            last_line = offset + used
        elif BOXED_RE.search(content) and not _balanced(content):
            rest = "\n".join([content] + lines[i + 1:])
            spans = _boxed_spans(rest)
            if not spans:
                return None
            content, end = spans[0]
            last_line = i + rest[:end].count("\n")

        if token_limit and last_line >= len(lines) - 1 and not text.endswith("\n"):
            return None
        if BOXED_RE.search(content):
            spans = _boxed_spans(content)
            content = spans[-1][0] if spans else content
        return normalize_answer(content)

    if token_limit:
        return None
    spans = _boxed_spans(text)
    return normalize_answer(spans[-1][0]) if spans else None
