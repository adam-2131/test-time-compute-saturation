"""Post-hoc rescoring of the three original runs from the response cache.

This is rescoring, not a new run. For every problem and every k the original
k_used is kept, so each vote is taken over exactly the responses the original
run drew, in the original order. Only answer extraction, normalization and
scoring change (see parse.py). The early-stopping decisions are the
original ones and were made with the original extractor.

The script never contacts the OpenAI API: it does not import openai or
utils.py, it reconstructs the cache keys of utils.py itself, and it disables
network sockets before reading anything. A missing cache entry is an error.

Before rescoring, every original CSV row is reproduced from the cache with the
original extractor and the original stopping rule, which verifies that the
reconstructed cache keys and response prefixes are the ones the run used.

    python scripts/replay_corrected.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import socket
import subprocess
import sys
from collections import Counter
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from parse import extract_answer, normalize_gold  # noqa: E402
from data import load_jsonl_dataset  # noqa: E402
from prompting import format_math_prompt  # noqa: E402

RUNS = {
    "aime_4o": {"model": "gpt-4o-mini", "dataset": "data/aime.jsonl", "original_dir": "experiments"},
    "gsm8k_4o": {"model": "gpt-4o-mini", "dataset": "data/gsm8k.jsonl", "original_dir": "experiments_gsm8k"},
    "aime_41": {"model": "gpt-4.1-mini", "dataset": "data/aime.jsonl", "original_dir": "experiments_aime_41mini"},
}
K_VALUES = [1, 10, 100, 1000]
PAIRS = [(1, 10), (10, 100), (100, 1000)]

TEMPERATURE = 0.7
MAX_TOKENS = 2048
BASE_SEED = 0
SEED_TAG = "fixed_k"
SUBSET_SIZE = 300
SUBSET_SEED = 123
MIN_SAMPLES = 10
STABILITY = 50
CONFIDENCE = 0.95

ORIGINAL_COLUMNS = ["example_id", "k_max", "k_used", "prediction", "confidence",
                    "entropy", "correct", "runtime_seconds"]
CORRECTED_COLUMNS = ["example_id", "k_max", "k_used", "prediction", "confidence", "entropy",
                     "correct", "valid_votes", "invalid_votes", "token_limit_votes",
                     "token_limit_valid_votes", "original_prediction", "original_correct"]


class MissingCacheEntry(RuntimeError):
    pass


class VerificationError(RuntimeError):
    pass


def forbid_network() -> None:
    def refuse(*args, **kwargs):
        raise RuntimeError("network access is disabled during offline rescoring")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse


# --- reconstructed from the code that produced the original CSVs (HEAD 52a9171) ---

def derive_seed(problem_index: int, call_index: int) -> int:
    data = f"{BASE_SEED}:{problem_index}:{call_index}:{SEED_TAG}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big") & 0x7FFFFFFF


def cache_file(cache_dir: Path, model: str, prompt: str, seed: int) -> Path:
    request = {"model": model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
               "prompt": prompt, "seed": seed}
    key = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def select_subset(problems: list[dict]) -> list[dict]:
    if len(problems) <= SUBSET_SIZE:
        return problems
    indices = list(range(len(problems)))
    random.Random(SUBSET_SEED).shuffle(indices)
    return [problems[i] for i in sorted(indices[:SUBSET_SIZE])]


_ORIGINAL_PATTERNS = [
    r"\\boxed\{([^}]+)\}",
    r"(?i)final\s*answer\s*[:=]\s*([^\n]+)",
    r"(?i)answer\s*[:=]\s*([^\n]+)",
]
_ORIGINAL_WRAPPERS = "${}\\ \t"


def original_extract(text: str) -> str | None:
    """parse.extract_final_answer as committed; used only to verify the replay."""
    if not text:
        return None
    for pattern in _ORIGINAL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            s = match.group(1).replace("\\boxed", "").replace("\\text", "")
            s = s.strip().strip(_ORIGINAL_WRAPPERS)
            s = s.strip().strip(".")
            return s.strip().strip(_ORIGINAL_WRAPPERS).strip()
    numbers = re.findall(r"(?<!\d)-?\d+(?!\d)", text)
    return numbers[-1] if numbers else None


def majority_stats(answers: list[str | None]) -> tuple[str | None, float, float]:
    """Plurality vote exactly as in the original eval.py (ties: smallest string)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0, 0.0
    counts = Counter(valid)
    total = len(valid)
    top = max(counts.values())
    prediction = sorted(a for a, c in counts.items() if c == top)[0]
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return prediction, top / total, entropy


def original_k_used(answers: list[str | None], k_max: int) -> int:
    """How many samples the original stopping rule drew, given the original answers."""
    min_samples = max(1, min(MIN_SAMPLES, k_max))
    previous, stable = None, 0
    for k_used in range(1, k_max + 1):
        if k_used > len(answers):
            raise MissingCacheEntry(f"stopping rule needs sample {k_used}, only {len(answers)} cached")
        if k_used < min_samples:
            continue
        majority, confidence, _ = majority_stats(answers[:k_used])
        if confidence >= CONFIDENCE:
            return k_used
        if majority is not None and majority == previous:
            stable += 1
            if stable >= STABILITY:
                return k_used
        else:
            stable = 1
        previous = majority
    return k_max


# --- rescoring ---

def mcnemar_exact(lost: int, won: int) -> float:
    discordant = lost + won
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(lost, won) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_original(run: str, k: int) -> list[dict[str, str]]:
    path = REPO_ROOT / RUNS[run]["original_dir"] / f"k_{k}.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or list(rows[0]) != ORIGINAL_COLUMNS:
        raise VerificationError(f"{path}: unexpected columns")
    return rows


def load_responses(cache_dir: Path, model: str, problem_index: int, prompt: str,
                   count: int, example_id: str) -> list[dict]:
    responses = []
    for call_index in range(count):
        path = cache_file(cache_dir, model, prompt, derive_seed(problem_index, call_index))
        if not path.is_file():
            raise MissingCacheEntry(
                f"{model} problem {example_id} (index {problem_index}) call {call_index}: {path.name} not in cache")
        entry = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entry.get("text"), str) or "completion_tokens" not in entry:
            raise VerificationError(f"{path.name}: malformed cache entry")
        responses.append({"text": entry["text"], "completion_tokens": entry["completion_tokens"],
                          "file": path.name})
    return responses


def rescore_run(run: str, cache_dir: Path) -> dict:
    cfg = RUNS[run]
    problems = select_subset(load_jsonl_dataset(REPO_ROOT / cfg["dataset"]))
    originals = {k: read_original(run, k) for k in K_VALUES}

    ids = [p["id"] for p in problems]
    for k, rows in originals.items():
        if [r["example_id"] for r in rows] != ids:
            raise VerificationError(f"{run} k={k}: problem ids or order differ from the subset")
    if len(set(ids)) != SUBSET_SIZE:
        raise VerificationError(f"{run}: expected {SUBSET_SIZE} distinct problems, got {len(set(ids))}")

    cache_files: set[str] = set()
    token_limit_rows = []
    corrected = {k: [] for k in K_VALUES}

    for index, problem in enumerate(problems):
        prompt = format_math_prompt(problem["problem"])
        needed = max(int(originals[k][index]["k_used"]) for k in K_VALUES)
        responses = load_responses(cache_dir, cfg["model"], index, prompt, needed, problem["id"])
        cache_files.update(r["file"] for r in responses)

        original_answers = [original_extract(r["text"]) for r in responses]
        flags = [(r["completion_tokens"] or 0) >= MAX_TOKENS for r in responses]
        answers = [extract_answer(r["text"], token_limit=f) for r, f in zip(responses, flags)]
        gold = normalize_gold(problem["answer"])
        if not gold:
            raise VerificationError(f"{run} {problem['id']}: gold answer {problem['answer']!r} not normalizable")

        for call_index, (r, flag, answer) in enumerate(zip(responses, flags, answers)):
            if flag:
                token_limit_rows.append({
                    "run": run, "example_id": problem["id"], "call_index": call_index,
                    "completion_tokens": r["completion_tokens"],
                    "corrected_answer": "" if answer is None else answer,
                    "original_answer": "" if original_answers[call_index] is None else original_answers[call_index],
                })

        for k in K_VALUES:
            row = originals[k][index]
            k_used = int(row["k_used"])

            # Verify: the original extractor and stopping rule reproduce the original row.
            if original_k_used(original_answers, k) != k_used:
                raise VerificationError(f"{run} k={k} {problem['id']}: original k_used not reproduced")
            pred, conf, ent = majority_stats(original_answers[:k_used])
            reproduced = {
                "prediction": "" if pred is None else pred,
                "confidence": f"{conf:.6f}",
                "entropy": f"{ent:.6f}",
                "correct": str(int(pred is not None and pred == problem["answer"].strip())),
            }
            for field, value in reproduced.items():
                if row[field] != value:
                    raise VerificationError(
                        f"{run} k={k} {problem['id']}: original {field} {row[field]!r} not reproduced ({value!r})")

            used = answers[:k_used]
            used_flags = flags[:k_used]
            pred, conf, ent = majority_stats(used)
            corrected[k].append({
                "example_id": problem["id"],
                "k_max": k,
                "k_used": k_used,
                "prediction": "" if pred is None else pred,
                "confidence": f"{conf:.6f}",
                "entropy": f"{ent:.6f}",
                "correct": int(pred is not None and pred in gold),
                "valid_votes": sum(a is not None for a in used),
                "invalid_votes": sum(a is None for a in used),
                "token_limit_votes": sum(used_flags),
                "token_limit_valid_votes": sum(f and a is not None for f, a in zip(used_flags, used)),
                "original_prediction": row["prediction"],
                "original_correct": int(row["correct"]),
                "_original_invalid": sum(a is None for a in original_answers[:k_used]),
                "_original_confidence": float(row["confidence"]),
                "_original_entropy": float(row["entropy"]),
                "_unit_only_mismatch": int(pred is not None and pred not in gold
                                           and re.sub(r"^\$|%$| [a-z ]+(\^\{?\d\}?)?$|°$", "", pred) in gold),
            })

    return {"ids": ids, "corrected": corrected, "token_limit_rows": token_limit_rows,
            "cache_files": cache_files}


def csv_text(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def summarize(run: str, result: dict) -> tuple[list[dict], list[dict]]:
    summary, tests = [], []
    for k in K_VALUES:
        rows = result["corrected"][k]
        n = len(rows)
        votes = sum(r["k_used"] for r in rows)
        with_prediction = [r for r in rows if r["prediction"] != ""]
        summary.append({
            "run": run, "k": k, "problems": n,
            "accuracy_original": sum(r["original_correct"] for r in rows) / n,
            "accuracy_corrected": sum(r["correct"] for r in rows) / n,
            "avg_k_used": votes / n,
            "avg_confidence_original": sum(r["_original_confidence"] for r in rows) / n,
            "avg_confidence_corrected": sum(float(r["confidence"]) for r in rows) / n,
            "avg_entropy_original": sum(r["_original_entropy"] for r in rows) / n,
            "avg_entropy_corrected": sum(float(r["entropy"]) for r in rows) / n,
            "responses_used": votes,
            "invalid_responses_original": sum(r["_original_invalid"] for r in rows),
            "invalid_responses_corrected": sum(r["invalid_votes"] for r in rows),
            "token_limit_responses": sum(r["token_limit_votes"] for r in rows),
            "token_limit_rate": sum(r["token_limit_votes"] for r in rows) / votes,
            "token_limit_responses_with_valid_answer": sum(r["token_limit_valid_votes"] for r in rows),
            "problems_without_valid_prediction": n - len(with_prediction),
            "accuracy_among_valid_predictions": (
                sum(r["correct"] for r in with_prediction) / len(with_prediction) if with_prediction else 0.0),
            "unit_only_mismatches": sum(r["_unit_only_mismatch"] for r in rows),
        })
    by_k = {k: {r["example_id"]: r for r in result["corrected"][k]} for k in K_VALUES}
    for lo, hi in PAIRS:
        if list(by_k[lo]) != list(by_k[hi]) or list(by_k[lo]) != result["ids"]:
            raise VerificationError(f"{run}: McNemar pair {lo}/{hi} not on the same problem ids")
        for label, field in (("original", "original_correct"), ("corrected", "correct")):
            won = sum(not by_k[lo][i][field] and by_k[hi][i][field] for i in result["ids"])
            lost = sum(by_k[lo][i][field] and not by_k[hi][i][field] for i in result["ids"])
            tests.append({"run": run, "comparison": f"k={lo} vs k={hi}", "scoring": label,
                          "problems": len(result["ids"]), "wrong_to_right": won, "right_to_wrong": lost,
                          "p_value": mcnemar_exact(lost, won)})
    return summary, tests


def git_blob_matches(path: Path) -> bool | None:
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
        head = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        work = subprocess.run(["git", "hash-object", str(path)], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        return head == work
    except (OSError, subprocess.CalledProcessError):
        return None


def report_markdown(summary: list[dict], tests: list[dict], totals: dict) -> str:
    pct = lambda x: f"{100 * x:.2f} %"
    lines = [
        "# Corrected rescoring of the original experiments",
        "",
        "**This is post-hoc rescoring, not a new run.** For every problem and k the original",
        "`k_used` is kept, so each vote uses exactly the responses the original run drew, in the",
        "original order. Only answer extraction, normalization and scoring were changed",
        "(`parse.py`). The early-stopping decisions are the original ones and were",
        "made with the original extractor; the corrected stopping rule was **not** rerun.",
        "",
        f"No API calls: network sockets were disabled, and all {totals['cache_files']} cache entries",
        "used were read directly and verified. Every original CSV row was first reproduced exactly",
        "from the cache with the original extractor and stopping rule.",
        "",
        "## Accuracy",
        "",
        "Accuracy is over all 300 problems; a problem without a valid prediction counts as wrong.",
        "The last column restricts accuracy to problems that have a valid prediction. That subset",
        "differs between k and between runs, so it describes the data and is not a comparable score.",
        "",
        "| Run | k | Original | Corrected | Avg. k_used | Problems without valid prediction | Corrected, among valid predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(f"| {s['run']} | {s['k']} | {pct(s['accuracy_original'])} | {pct(s['accuracy_corrected'])} "
                     f"| {s['avg_k_used']:.2f} | {s['problems_without_valid_prediction']} "
                     f"| {pct(s['accuracy_among_valid_predictions'])} |")
    lines += ["", "## Confidence and entropy", "",
              "| Run | k | Confidence original | Confidence corrected | Entropy original (bits) | Entropy corrected (bits) |",
              "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['run']} | {s['k']} | {s['avg_confidence_original']:.3f} | {s['avg_confidence_corrected']:.3f} "
                     f"| {s['avg_entropy_original']:.3f} | {s['avg_entropy_corrected']:.3f} |")
    lines += ["", "## Invalid and token-limit responses", "",
              "Counts are over the responses that enter the votes (the first `k_used` responses of each problem).",
              "",
              "| Run | k | Responses | Invalid (original) | Invalid (corrected) | Token-limit | Token-limit rate | Token-limit with valid answer |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['run']} | {s['k']} | {s['responses_used']} | {s['invalid_responses_original']} "
                     f"| {s['invalid_responses_corrected']} | {s['token_limit_responses']} | {pct(s['token_limit_rate'])} "
                     f"| {s['token_limit_responses_with_valid_answer']} |")
    lines += ["", "## McNemar tests (paired, exact, two-sided)", "",
              "| Run | Comparison | Scoring | Wrong → right | Right → wrong | p |",
              "|---|---|---|---:|---:|---:|"]
    for t in tests:
        lines.append(f"| {t['run']} | {t['comparison']} | {t['scoring']} | {t['wrong_to_right']} "
                     f"| {t['right_to_wrong']} | {t['p_value']:.4f} |")
    lines += ["", "## Scoring rules", "",
              "- An answer counts as correct only if its canonical form equals the canonical gold answer.",
              "- Percent signs, currency signs, degree signs and unit words are kept, so `50%` is not `50`.",
              "  `unit_only_mismatches` in `comparison.csv` counts final predictions that differ from the gold",
              "  answer only by such a marker; they are scored as wrong and are reported, not reinterpreted.",
              "- Responses that hit the 2048-token limit keep an answer only if they contain a complete,",
              "  explicit `Final Answer:`; they are listed individually in `token_limit_responses.csv`.",
              "- Ties in the plurality vote are broken as in the original run (smallest string).",
              ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline corrected rescoring of the original runs")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "corrected_results")
    args = parser.parse_args()

    forbid_network()

    originals = sorted((REPO_ROOT / cfg["original_dir"] / f"k_{k}.csv")
                       for cfg in RUNS.values() for k in K_VALUES)
    hashes_before = {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in originals}

    results, summary, tests, token_rows, cache_files = {}, [], [], [], set()
    for run in RUNS:
        print(f"{run}: verifying and rescoring ...", flush=True)
        results[run] = rescore_run(run, args.cache_dir)
        s, t = summarize(run, results[run])
        summary += s
        tests += t
        token_rows += results[run]["token_limit_rows"]
        cache_files |= results[run]["cache_files"]

    outputs: dict[str, str] = {}
    for run, result in results.items():
        for k in K_VALUES:
            outputs[f"{run}/k_{k}.csv"] = csv_text(result["corrected"][k], CORRECTED_COLUMNS)
    outputs["token_limit_responses.csv"] = csv_text(
        token_rows, ["run", "example_id", "call_index", "completion_tokens", "corrected_answer", "original_answer"])
    outputs["comparison.csv"] = csv_text(
        [{**s, **{f: f"{s[f]:.6f}" for f in s if isinstance(s[f], float)}} for s in summary], list(summary[0]))
    outputs["mcnemar.csv"] = csv_text([{**t, "p_value": f"{t['p_value']:.6f}"} for t in tests], list(tests[0]))
    outputs["comparison_report.md"] = report_markdown(summary, tests, {"cache_files": len(cache_files)})

    hashes_after = {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in originals}
    if hashes_after != hashes_before:
        raise VerificationError("an original CSV changed during rescoring")
    if "openai" in sys.modules or "utils" in sys.modules:
        raise VerificationError("the OpenAI client was imported")

    manifest = {
        "description": "post-hoc rescoring with original k_used; stopping rule not rerun",
        "network_disabled": True,
        "openai_imported": False,
        "settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "base_seed": BASE_SEED,
                     "seed_tag": SEED_TAG, "subset_size": SUBSET_SIZE, "subset_seed": SUBSET_SEED},
        "runs": {run: {**cfg, "problems": len(results[run]["ids"]),
                       "cache_entries_verified": len(results[run]["cache_files"])}
                 for run, cfg in RUNS.items()},
        "cache_entries_verified": len(cache_files),
        "original_rows_reproduced": sum(len(r["corrected"][k]) for r in results.values() for k in K_VALUES),
        "original_csv_sha256": hashes_before,
        "inputs_match_git_head": {f: git_blob_matches(REPO_ROOT / f)
                                  for f in ["prompting.py", "data.py", *hashes_before]},
        "code_sha256": {f: sha256(REPO_ROOT / f) for f in ["parse.py", "scripts/replay_corrected.py"]},
        "output_sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                          for name, text in sorted(outputs.items())},
    }
    outputs["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    for name, text in outputs.items():
        path = args.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    print(f"verified {len(cache_files)} cache entries and "
          f"{manifest['original_rows_reproduced']} original rows; wrote {len(outputs)} files to {args.out_dir}")
    print()
    print(f"{'run':<10}{'k':>6}{'original':>10}{'corrected':>11}{'k_used':>9}{'no pred':>9}")
    for s in summary:
        print(f"{s['run']:<10}{s['k']:>6}{s['accuracy_original']:>10.4f}{s['accuracy_corrected']:>11.4f}"
              f"{s['avg_k_used']:>9.2f}{s['problems_without_valid_prediction']:>9}")


if __name__ == "__main__":
    main()
