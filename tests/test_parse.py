"""Answer extraction. A change here silently moves every accuracy number."""

import pytest

from parse import extract_final_answer, normalize


class TestPatternPriority:
    def test_boxed(self):
        assert extract_final_answer(r"so \boxed{106}") == "106"

    def test_final_answer_line(self):
        assert extract_final_answer("Final Answer: 42") == "42"

    def test_bare_answer_line(self):
        assert extract_final_answer("Answer: 42") == "42"

    def test_answer_with_equals(self):
        assert extract_final_answer("Answer = 42") == "42"

    @pytest.mark.parametrize("text", ["FINAL ANSWER: 42", "final answer: 42"])
    def test_is_case_insensitive(self, text):
        assert extract_final_answer(text) == "42"

    def test_boxed_wins_over_a_later_final_answer_line(self):
        text = r"\boxed{7}" "\nFinal Answer: 9"
        assert extract_final_answer(text) == "7"

    def test_final_answer_wins_over_the_trailing_integer_fallback(self):
        assert extract_final_answer("Final Answer: 42\nchecked against 99") == "42"


class TestFallback:
    def test_uses_the_last_integer_when_no_pattern_matches(self):
        assert extract_final_answer("I tried 3, then 12, and got 99") == "99"

    def test_keeps_a_negative_sign(self):
        assert extract_final_answer("the value is -5") == "-5"

    def test_returns_none_without_digits(self):
        assert extract_final_answer("I could not solve this.") is None

    @pytest.mark.parametrize("text", ["", None])
    def test_returns_none_for_empty_input(self, text):
        assert extract_final_answer(text) is None


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("106", "106"),
            ("  106  ", "106"),
            ("$106$", "106"),
            ("{106}", "106"),
            ("106.", "106"),
            ("} 106", "106"),
            (r"\boxed{106}", "106"),
            (r"\text{106}", "106"),
        ],
    )
    def test_strips_latex_and_markdown_wrappers(self, raw, expected):
        assert normalize(raw) == expected

    def test_keeps_a_negative_sign(self):
        assert normalize("$-5$") == "-5"

    def test_leaves_interior_characters_alone(self):
        assert normalize("3/4") == "3/4"


class TestExtractedAnswersCompareEqualToGold:
    """Gold answers are compared as plain strings, so wrappers must be gone."""

    @pytest.mark.parametrize(
        "text",
        [
            r"reasoning \boxed{106}",
            "Final Answer: 106",
            "Final Answer: $106$",
            "Final Answer: 106.",
            r"Final Answer: \boxed{106}",
        ],
    )
    def test_all_forms_reduce_to_the_same_string(self, text):
        assert extract_final_answer(text) == "106"
