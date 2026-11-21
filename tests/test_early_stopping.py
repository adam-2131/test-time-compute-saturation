
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
    def test_unanimous_stops_at_min_samples(self, monkeypatch):
        stub(monkeypatch, ["7"])
        result = run(k_max=100)

        assert result["k_used"] == 10
        assert result["confidence"] == 1.0
        assert result["correct"] is True

    @pytest.mark.parametrize("k_max", [1, 2, 4, 9])
    def test_k_max_below_min_samples(self, monkeypatch, k_max):
        stub(monkeypatch, ["7"])
        assert run(k_max=k_max, early_stop_min_samples=10)["k_used"] == k_max

    def test_single_sample(self, monkeypatch):
        stub(monkeypatch, ["7"])
        result = run(k_max=1)

        assert result["k_used"] == 1
        assert result["confidence"] == 1.0
        assert result["entropy"] == 0.0

    def test_mixed_votes_keep_sampling(self, monkeypatch):
        stub(monkeypatch, ["7", "7", "7", "9"])

        result = run(k_max=20, early_stop_stability=10_000)
        assert result["k_used"] == 20


class TestStabilityCriterion:
    def test_stable_majority_stops(self, monkeypatch):
        stub(monkeypatch, ["7", "7", "9"])
        result = run(k_max=100, early_stop_stability=3)

        assert result["k_used"] == 12
        assert result["prediction"] == "7"
        assert result["confidence"] == pytest.approx(8 / 12)

    def test_larger_stability_window(self, monkeypatch):
        stub(monkeypatch, ["7", "7", "9"])
        assert run(k_max=100, early_stop_stability=8)["k_used"] == 17


class TestUnparseableAnswers:
    def test_no_valid_answers(self, monkeypatch):
        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            return ModelOutput(
                text="no idea",
                completion_tokens=5,
                prompt_tokens=5,
                total_tokens=10,
            )

        monkeypatch.setattr(ev, "call_model", fake)
        result = run(k_max=12, early_stop_stability=3)

        assert result["prediction"] is None
        assert result["correct"] is False
        assert result["k_used"] == 12
        assert result["valid_votes"] == 0
        assert result["invalid_votes"] == 12


def stub_texts(monkeypatch, texts, **output_fields):
    calls = itertools.count()

    def fake(prompt, *, model, temperature, max_tokens, seed=None):
        fields = dict(completion_tokens=10, prompt_tokens=5, total_tokens=15)
        fields.update(output_fields)
        return ModelOutput(text=texts[min(next(calls), len(texts) - 1)], **fields)

    monkeypatch.setattr(ev, "call_model", fake)


class TestInvalidResponsesDoNotDriveStopping:
    def test_one_valid_answer_among_invalid_ones_does_not_stop(self, monkeypatch):
        # confidence is 1/1 = 100 %, but only one response has an answer
        stub_texts(monkeypatch, ["Final Answer: 7"] + ["no idea"] * 99)
        result = run(k_max=40, early_stop_stability=10_000)

        assert result["k_used"] == 40
        assert result["confidence"] == 1.0
        assert result["valid_votes"] == 1
        assert result["invalid_votes"] == 39

    def test_confidence_stop_waits_for_enough_valid_answers(self, monkeypatch):
        texts = ["no idea"] * 10 + ["Final Answer: 7"] * 50
        stub_texts(monkeypatch, texts)
        result = run(k_max=100, early_stop_min_valid=10)

        assert result["k_used"] == 20
        assert result["valid_votes"] == 10

    def test_invalid_responses_do_not_count_as_stable(self, monkeypatch):
        texts = ["Final Answer: 7", "Final Answer: 9"] * 6 + ["no idea"] * 200
        stub_texts(monkeypatch, texts)
        # the majority is stable for 3 valid answers; 2 more invalid ones
        # would have reached the window of 5 under the old rule
        result = run(k_max=100, early_stop_stability=5, early_stop_min_valid=1)

        assert result["k_used"] == 100
        assert result["valid_votes"] == 12

    def test_all_valid_answers_behave_as_before(self, monkeypatch):
        stub(monkeypatch, ["7", "7", "9"])
        assert run(k_max=100, early_stop_stability=3)["k_used"] == 12


class TestTokenLimit:
    def test_finish_reason_length_marks_a_response_as_cut_off(self, monkeypatch):
        stub_texts(monkeypatch, ["working...\nFinal Answer: 7"], finish_reason="length")
        result = run(k_max=1)

        assert result["token_limit_votes"] == 1
        assert result["prediction"] is None

    def test_a_complete_final_answer_survives_the_token_limit(self, monkeypatch):
        stub_texts(monkeypatch, ["working...\nFinal Answer: 7\nmore"], finish_reason="length")
        result = run(k_max=1)

        assert result["token_limit_votes"] == 1
        assert result["prediction"] == "7"

    def test_old_cache_entries_use_the_token_count(self, monkeypatch):
        stub_texts(monkeypatch, ["working...\nFinal Answer: 7"], completion_tokens=2048)
        assert run(k_max=1)["token_limit_votes"] == 1

    def test_finish_reason_stop_is_not_cut_off(self, monkeypatch):
        stub_texts(monkeypatch, ["Final Answer: 7"], completion_tokens=2048, finish_reason="stop")
        result = run(k_max=1)

        assert result["token_limit_votes"] == 0
        assert result["prediction"] == "7"

    def test_completion_tokens_are_summed(self, monkeypatch):
        stub(monkeypatch, ["7"])
        assert run(k_max=3, early_stop_min_samples=3)["completion_tokens"] == 30


class TestOrderingAndConcurrency:
    @pytest.mark.parametrize("max_workers", [1, 4])
    def test_input_order_preserved(self, monkeypatch, max_workers):
        problems = [
            {"id": f"p{i}", "problem": f"problem {i}", "answer": str(i)}
            for i in range(12)
        ]

        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            return reply(prompt.rsplit("problem ", 1)[1].strip())

        monkeypatch.setattr(ev, "call_model", fake)

        results = ev.run_fixed_k_majority(
            problems, "m", 0.7, 2048,
            k_max=3, seed=0, max_workers=max_workers
        )

        assert [r["example_id"] for r in results] == [
            f"p{i}" for i in range(12)
        ]
        assert all(r["correct"] for r in results)


class TestRetries:
    def test_fatal_error_no_retry(self, monkeypatch):
        calls = itertools.count()

        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            next(calls)
            raise FatalModelError("OPENAI_API_KEY was rejected (HTTP 401)")

        monkeypatch.setattr(ev, "call_model", fake)
        monkeypatch.setattr(ev.time, "sleep", lambda *_: None)

        with pytest.raises(FatalModelError):
            run(k_max=5)

        assert next(calls) == 1, "a rejected key must not be retried"

    def test_transient_error_retries(self, monkeypatch):
        attempts = itertools.count()

        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            if next(attempts) < 2:
                raise ConnectionError("transient")
            return reply("7")

        monkeypatch.setattr(ev, "call_model", fake)
        monkeypatch.setattr(ev.time, "sleep", lambda *_: None)

        assert run(k_max=1)["correct"] is True

    def test_retry_limit(self, monkeypatch):
        def fake(prompt, *, model, temperature, max_tokens, seed=None):
            raise ConnectionError("transient")

        monkeypatch.setattr(ev, "call_model", fake)
        monkeypatch.setattr(ev.time, "sleep", lambda *_: None)

        with pytest.raises(RuntimeError, match="failed after repeated retries"):
            run(k_max=1)
