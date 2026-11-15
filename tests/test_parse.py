
import pytest

from parse import extract_final_answer, normalize


class TestPatternPriority:
    def test_boxed(self):
        assert extract_final_answer(r"so \boxed{106}") == "106"

    def test_final_answer(self):
        assert extract_final_answer("Final Answer: 42") == "42"

    def test_answer_line(self):
        assert extract_final_answer("Answer: 42") == "42"

    def test_answer_equals(self):
        assert extract_final_answer("Answer = 42") == "42"

    @pytest.mark.parametrize("text", ["FINAL ANSWER: 42", "final answer: 42"])
    def test_case_insensitive(self, text):
        assert extract_final_answer(text) == "42"

    def test_boxed_priority(self):
        text = r"\boxed{7}" "\nFinal Answer: 9"
        assert extract_final_answer(text) == "7"

    def test_final_answer_priority(self):
        assert extract_final_answer("Final Answer: 42\nchecked against 99") == "42"


class TestFallback:
    def test_last_integer(self):
        assert extract_final_answer("I tried 3, then 12, and got 99") == "99"

    def test_negative_integer(self):
        assert extract_final_answer("the value is -5") == "-5"

    def test_no_digits(self):
        assert extract_final_answer("I could not solve this.") is None

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_input(self, text):
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
    def test_wrappers(self, raw, expected):
        assert normalize(raw) == expected

    def test_negative(self):
        assert normalize("$-5$") == "-5"

    def test_fraction(self):
        assert normalize("3/4") == "3/4"


class TestExtractedAnswersCompareEqualToGold:
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
    def test_matches_gold(self, text):
        assert extract_final_answer(text) == "106"
