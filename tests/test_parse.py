"""The conservative extractor used by the experiment and the post-hoc rescoring."""

import pytest

from parse import extract_answer, normalize_answer, normalize_gold


class TestFinalAnswerLabel:
    def test_bold_label_line_before_the_real_one(self):
        assert extract_answer("**Final Answer:**\n\nFinal Answer: 106") == "106"

    def test_bold_wrapped_label(self):
        assert extract_answer("**Final Answer: 35**") == "35"

    def test_heading_label_does_not_swallow_the_next_label(self):
        assert extract_answer("### Final Answer:\n\nFinal Answer: 26") == "26"

    def test_the_last_label_wins(self):
        text = "\\[\n\\text{Final Answer: } 26 \\text{ and } 74\n\\]\nreport the smaller one:\n\nFinal Answer: 26"
        assert extract_answer(text) == "26"

    def test_label_beats_an_earlier_intermediate_box(self):
        text = "so \\boxed{\\frac{7}{99}} is the probability\n\nFinal Answer: 106"
        assert extract_answer(text) == "106"

    def test_label_beats_a_box_with_a_unit(self):
        assert extract_answer("\\[\n\\boxed{160^\\circ}\n\\]\n\nFinal Answer: 160") == "160"

    def test_answer_on_the_lines_after_an_empty_label(self):
        assert extract_answer("### Final Answer:\n\n\\[\n\\boxed{26}\n\\]\n") == "26"

    def test_box_spanning_several_lines_after_the_label(self):
        assert extract_answer("Final Answer:\n\\boxed{\n  42\n}\n") == "42"

    def test_label_inside_a_box(self):
        assert extract_answer("### Final Answer:\n\n\\[\n\\boxed{\\text{Final Answer: } 955}\n\\]") == "955"

    def test_label_inside_display_math(self):
        assert extract_answer("\\[ \\text{Final Answer: } 200 \\]") == "200"

    def test_symbolic_assignment_keeps_the_value(self):
        assert extract_answer("Final Answer: m + n = 106") == "106"

    def test_prose_after_the_label_is_not_an_answer(self):
        text = "Final Answer: 0 (not enough valid arrangements exist to meet the counts)"
        assert extract_answer(text) is None

    def test_hedged_answer_is_not_an_answer(self):
        assert extract_answer("Final Answer: \\(ab \\approx 118.27\\)") is None

    def test_two_answers_are_not_an_answer(self):
        assert extract_answer("Final Answer: 26 or 74") is None


class TestWithoutLabel:
    def test_last_complete_box_is_used(self):
        assert extract_answer("first \\boxed{3}, then corrected to \\boxed{\\frac{1}{2}}") == "1/2"

    def test_nested_braces_are_balanced(self):
        assert extract_answer("\\boxed{86 - 12\\sqrt{10}}") == "86 - 12\\sqrt{10}"

    def test_no_last_integer_fallback(self):
        assert extract_answer("I tried 3, then 12, and the value is 99") is None

    def test_incomplete_box_is_not_an_answer(self):
        assert extract_answer("so the answer is \\boxed{12") is None

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_response(self, text):
        assert extract_answer(text) is None


class TestTokenLimit:
    def test_unfinished_reasoning_gives_no_answer(self):
        assert extract_answer("992^2 = 1,000,000 - 16,000 +", token_limit=True) is None

    def test_a_box_is_not_trusted_without_a_label(self):
        assert extract_answer("so \\boxed{45} for now, continuing", token_limit=True) is None

    def test_a_complete_label_line_is_kept(self):
        assert extract_answer("Final Answer: 45\nLet me double-check this", token_limit=True) == "45"

    def test_a_label_on_the_cut_off_last_line_is_dropped(self):
        assert extract_answer("work\nFinal Answer: 4", token_limit=True) is None

    def test_the_same_text_is_kept_without_the_flag(self):
        assert extract_answer("work\nFinal Answer: 4") == "4"


class TestNormalization:
    @pytest.mark.parametrize("raw, expected", [
        ("025", "25"), ("008", "8"),
        ("4.00", "4"), ("4.0", "4"), ("0.50", "0.5"), ("7425.00", "7425"),
        ("-5", "-5"), ("−5", "-5"), ("+5", "5"), ("-0.0", "0"),
        ("1,000", "1000"), ("12.5", "12.5"),
        ("$106$", "106"), ("\\(106\\)", "106"), ("\\boxed{106}", "106"), ("106.", "106"),
        ("\\frac{3}{4}", "3/4"), ("-\\frac{1}{2}", "-1/2"), ("6/8", "6/8"),
    ])
    def test_exact_representations(self, raw, expected):
        assert normalize_answer(raw) == expected

    @pytest.mark.parametrize("raw", ["50%", "50\\%", "50 percent"])
    def test_percent_is_kept(self, raw):
        assert normalize_answer(raw) == "50%"

    def test_percent_is_not_the_plain_number(self):
        assert normalize_answer("50%") != normalize_answer("50")
        assert extract_answer("Final Answer: 50%") not in normalize_gold("50")

    def test_currency_and_units_are_kept(self):
        assert normalize_answer("$18") == "$18"
        assert normalize_answer("16 liters") == "16 liters"
        assert normalize_answer("160^\\circ") == "160°"

    def test_no_rounding(self):
        assert normalize_answer("168.75") == "168.75"
        assert normalize_answer("0.333333") == "0.333333"

    def test_fractions_are_not_converted_to_decimals(self):
        assert normalize_answer("\\frac{1}{2}") != normalize_answer("0.5")

    def test_normalization_is_idempotent(self):
        for raw in ["025", "4.00", "50%", "$18", "\\frac{3}{4}", "16 liters", "86 - 12\\sqrt{10}"]:
            once = normalize_answer(raw)
            assert normalize_answer(once) == once

    def test_unreliable_fragments_are_rejected(self):
        for raw in ["**", "[", "Final Answer:", "The square of the distance is:"]:
            assert normalize_answer(raw) is None


class TestGold:
    def test_integer_gold(self):
        assert normalize_gold(" 25 ") == {"25"}

    def test_dataset_marker_for_two_accepted_answers(self):
        assert normalize_gold("150 or 151 (both were accepted)") == {"150", "151"}
