from dataclasses import dataclass
from collections import Counter
from utils import call_model
from parse import extract_final_answer
from prompting import format_math_prompt

@dataclass
class Sample:
    text: str
    answer: str | None
    correct: bool
    completion_tokens: int

def greedy(problem: str, answer_key: str, *, model: str, max_tokens: int = 2048) -> Sample:
    out = call_model(format_math_prompt(problem), model=model, temperature=0.0, max_tokens=max_tokens)
    ans = extract_final_answer(out.text)
    ctoks = out.completion_tokens or 0
    return Sample(out.text, ans, ans == answer_key, ctoks)

def sample_n(problem: str, answer_key: str, *, model: str, n: int, temperature: float = 0.7, max_tokens: int = 2048) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(n):
        out = call_model(format_math_prompt(problem), model=model, temperature=temperature, max_tokens=max_tokens)
        ans = extract_final_answer(out.text)
        ctoks = out.completion_tokens or 0
        samples.append(Sample(out.text, ans, ans == answer_key, ctoks))
    return samples

def pass_at_k(samples: list[Sample]) -> bool:
    return any(s.correct for s in samples)

def majority_vote(samples: list[Sample]) -> str | None:
    answers = [s.answer for s in samples if s.answer is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]
