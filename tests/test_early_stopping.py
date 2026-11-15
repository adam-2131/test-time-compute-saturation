"""The early-stopping rules, which decide how much compute each k really uses."""

import itertools

import pytest

import eval as ev
from utils import FatalModelError, ModelOutput


def reply(answer):
    return ModelOutput(
        text=f"reasoning\nFinal Answer: {answer}",
        completion_tokens=10,
        prompt_tokens=5,
        total_tokens=15,
    )


def stub(monkeypatch, answers):
    """Serve `answers` in order, counting calls. Returns the counter."""
    calls = itertools.count()

    def fake(prompt, *, model, temperature, max_tokens, seed=None):
        i = next(calls)
        return reply(answers[i % len(answers)])

    monkeypatch.setattr(ev, "call_model", fake)
    return calls


ONE_PROBLEM = [{"id": "p0", "problem": "what is x?", "answer": "7"}]


def run(**kwargs):
    params = dict(seed=0, early_stop_confidence=0.95, early_stop_min_samples=10)
    params.update(kwargs)
    return ev.run_fixed_k_majority(ONE_PROBLEM, "m", 0.7, 2048, **params)[0]


class TestConfidenceCriterion:
    def test_unanimous_votes_stop_at_min_samples(self, monkeypatch):
        stub(monkeypatch, ["7"])
        result = run(k_max=100)
        # A unanimous vote is at confidence 1.0 from the first draw, so the
        # minimum is the only thing keeping it sampling.
        assert result["k_used"] == 10
        assert result["confidence"] == 1.0
        assert result["correct"] is True

    @pytest.mark.parametrize("k_max", [1, 2, 4, 9])
    def test_k_max_is_never_exceeded_below_min_samples(self, monkeypatch, k_max):
        stub(monkeypatch, ["7"])
        # min_samples sits above k_max here, so no criterion can fire and the
        # loop bound is the only thing stopping the run.
        assert run(k_max=k_max, early_stop_min_samples=10)["k_used"] == k_max

    def test_k_max_of_one_takes_a_single_sample(self, monkeypatch):
        stub(monkeypatch, ["7"])
        result = run(k_max=1)
        assert result["k_used"] == 1
        assert result["confidence"] == 1.0
        assert result["entropy"] == 0.0

    def test_a_lone_disagreement_does_not_stop_the_run_early(self, monkeypatch):
        # Two answers in 20 draws never reaches 0.95, so this runs to k_max.
        stub(monkeypatch, ["7", "7", "7", "9"])
        assert run(k_max=20, early_stop_stability=10_000)["k_used"] == 20


class TestStabilityCriterion:
    def test_stops_once_the_majority_has_held_long_enough(self, monkeypatch):
        # Repeating 7,7,9 keeps confidence near 0.67, so only stability can fire.
        stub(monkeypatch, ["7", "7", "9"])
        result = run(k_max=100, early_stop_stability=3)
        assert result["k_used"] == 12
        assert result["prediction"] == "7"
        assert result["confidence"] == pytest.approx(8 / 12)

    def test_a_higher_stability_threshold_samples_longer(self, monkeypatch):
        stub(monkeypatch, ["7", "7", "9"])
        assert run(k_max=100, early_stop_stability=8)["k_used"] == 17


class TestUnparseableAnswers:
    def test_a_run_with_no_parseable_answer_is_not_correct(self, monkeypatch):
        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            return ModelOutput(text="no idea", completion_tokens=5,
                               prompt_tokens=5, total_tokens=10)

        monkeypatch.setattr(ev, "call_model", fake)
        result = run(k_max=12, early_stop_stability=3)
        assert result["prediction"] is None
        assert result["correct"] is False
        # Nothing ever reaches a majority, so no criterion can fire.
        assert result["k_used"] == 12


class TestOrderingAndConcurrency:
    @pytest.mark.parametrize("max_workers", [1, 4])
    def test_results_come_back_in_input_order(self, monkeypatch, max_workers):
        problems = [
            {"id": f"p{i}", "problem": f"problem {i}", "answer": str(i)}
            for i in range(12)
        ]

        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            return reply(prompt.rsplit("problem ", 1)[1].strip())

        monkeypatch.setattr(ev, "call_model", fake)
        results = ev.run_fixed_k_majority(
            problems, "m", 0.7, 2048, k_max=3, seed=0, max_workers=max_workers
        )
        assert [r["example_id"] for r in results] == [f"p{i}" for i in range(12)]
        assert all(r["correct"] for r in results)


class TestRetries:
    def test_a_fatal_error_aborts_without_retrying(self, monkeypatch):
        calls = itertools.count()

        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            next(calls)
            raise FatalModelError("OPENAI_API_KEY was rejected (HTTP 401)")

        monkeypatch.setattr(ev, "call_model", fake)
        monkeypatch.setattr(ev.time, "sleep", lambda _: None)

        with pytest.raises(FatalModelError):
            run(k_max=5)
        assert next(calls) == 1, "a rejected key must not be retried"

    def test_a_transient_error_is_retried_at_the_problem_level(self, monkeypatch):
        attempts = itertools.count()

        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            if next(attempts) < 2:
                raise ConnectionError("transient")
            return reply("7")

        monkeypatch.setattr(ev, "call_model", fake)
        monkeypatch.setattr(ev.time, "sleep", lambda _: None)

        assert run(k_max=1)["correct"] is True

    def test_gives_up_after_repeated_failures(self, monkeypatch):
        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            raise ConnectionError("transient")

        monkeypatch.setattr(ev, "call_model", fake)
        monkeypatch.setattr(ev.time, "sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="failed after repeated retries"):
            run(k_max=1)
