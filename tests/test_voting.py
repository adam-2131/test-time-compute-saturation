"""Majority voting and seed derivation."""

import pytest

from eval import _derive_seed, majority_stats


class TestMajorityStats:
    def test_unanimous(self):
        prediction, confidence, entropy = majority_stats(["7", "7", "7"])
        assert prediction == "7"
        assert confidence == 1.0
        assert entropy == 0.0

    def test_clear_winner(self):
        prediction, confidence, _ = majority_stats(["7", "7", "9"])
        assert prediction == "7"
        assert confidence == pytest.approx(2 / 3)

    def test_even_split_has_one_bit_of_entropy(self):
        prediction, confidence, entropy = majority_stats(["a", "a", "b", "b"])
        assert confidence == 0.5
        assert entropy == pytest.approx(1.0)
        assert prediction == "a"

    def test_ties_break_lexicographically_not_by_insertion_order(self):
        assert majority_stats(["b", "a"])[0] == "a"
        assert majority_stats(["a", "b"])[0] == "a"

    def test_unparseable_answers_are_dropped_from_the_vote(self):
        prediction, confidence, _ = majority_stats([None, "7", None, "7"])
        assert prediction == "7"
        # Confidence is a share of the parsed votes, not of all attempts.
        assert confidence == 1.0

    def test_all_unparseable(self):
        assert majority_stats([None, None]) == (None, 0.0, 0.0)

    def test_empty(self):
        assert majority_stats([]) == (None, 0.0, 0.0)


class TestDeriveSeed:
    def test_is_deterministic(self):
        args = dict(problem_index=3, call_index=9, tag="fixed_k")
        assert _derive_seed(0, **args) == _derive_seed(0, **args)

    def test_stays_in_signed_32_bit_range(self):
        # The API rejects seeds outside this range.
        for call_index in range(200):
            seed = _derive_seed(7, problem_index=1, call_index=call_index, tag="t")
            assert 0 <= seed <= 0x7FFFFFFF

    @pytest.mark.parametrize(
        "changed",
        [
            dict(problem_index=2),
            dict(call_index=1),
            dict(tag="other"),
        ],
    )
    def test_varies_with_every_input(self, changed):
        base = dict(problem_index=1, call_index=0, tag="fixed_k")
        assert _derive_seed(0, **base) != _derive_seed(0, **{**base, **changed})

    def test_varies_with_base_seed(self):
        base = dict(problem_index=1, call_index=0, tag="fixed_k")
        assert _derive_seed(0, **base) != _derive_seed(1, **base)

    def test_call_indices_of_one_problem_rarely_collide(self):
        seeds = {
            _derive_seed(0, problem_index=0, call_index=i, tag="fixed_k")
            for i in range(1000)
        }
        assert len(seeds) == 1000

