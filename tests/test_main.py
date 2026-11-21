"""The command line passes its settings through to the experiment."""

import csv
import sys

import main


def run_main(monkeypatch, tmp_path, *extra):
    seen = {}

    def fake_run(problems, model, temperature, max_tokens, k_max, **kwargs):
        seen.update(problems=problems, model=model, max_tokens=max_tokens, **kwargs)
        return [
            {
                "example_id": p["id"], "k_max": k_max, "k_used": 1, "valid_votes": 1,
                "invalid_votes": 0, "token_limit_votes": 0, "prediction": "1",
                "confidence": 1.0, "entropy": 0.0, "correct": True,
                "completion_tokens": 5, "runtime_seconds": 0.1,
            }
            for p in problems
        ]

    monkeypatch.setattr(main, "run_fixed_k_majority", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--dataset-path", "data/aime.jsonl", "--k-values", "1",
        "--per-example-dir", str(tmp_path), *extra,
    ])
    main.main()
    return seen


def test_max_tokens_defaults_to_the_original_2048(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path)["max_tokens"] == 2048


def test_max_tokens_is_passed_through(monkeypatch, tmp_path):
    seen = run_main(monkeypatch, tmp_path, "--model", "gpt-4.1-mini", "--max-tokens", "16384")

    assert seen["max_tokens"] == 16384
    assert seen["model"] == "gpt-4.1-mini"


def test_limit_keeps_the_first_problems_of_the_full_subset(monkeypatch, tmp_path):
    full = run_main(monkeypatch, tmp_path)["problems"]
    pilot = run_main(monkeypatch, tmp_path, "--limit", "30")["problems"]

    assert len(full) == 300
    assert pilot == full[:30]


def test_min_valid_is_passed_through(monkeypatch, tmp_path):
    seen = run_main(monkeypatch, tmp_path, "--early-stop-min-valid", "5")
    assert seen["early_stop_min_valid"] == 5


def test_csv_has_the_vote_counts(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, "--limit", "3")

    with (tmp_path / "k_1.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert list(rows[0]) == main.COLUMNS
    assert {"valid_votes", "invalid_votes", "token_limit_votes"} <= set(rows[0])
    assert len(rows) == 3
