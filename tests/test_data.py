"""Dataset loading and subset selection, which fix what every run is scored on."""

import json

import pytest

from data import load_jsonl_dataset
from main import parse_int_list, select_subset


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


class TestLoadJsonlDataset:
    def test_reads_the_three_required_fields(self, tmp_path):
        path = write_jsonl(tmp_path / "d.jsonl", [
            {"id": "2009-II-9", "problem": "what is x?", "answer": "106"},
        ])
        assert load_jsonl_dataset(path) == [
            {"id": "2009-II-9", "problem": "what is x?", "answer": "106"},
        ]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            '{"id": "a", "problem": "p", "answer": "1"}\n'
            "\n"
            "   \n"
            '{"id": "b", "problem": "p", "answer": "2"}\n',
            encoding="utf-8",
        )
        assert [r["id"] for r in load_jsonl_dataset(path)] == ["a", "b"]

    def test_trims_gold_answers_because_they_are_compared_as_strings(self, tmp_path):
        path = write_jsonl(tmp_path / "d.jsonl", [
            {"id": "a", "problem": "p", "answer": "  106\n"},
        ])
        assert load_jsonl_dataset(path)[0]["answer"] == "106"

    def test_coerces_numeric_fields_to_strings(self, tmp_path):
        path = write_jsonl(tmp_path / "d.jsonl", [
            {"id": 7, "problem": "p", "answer": 106},
        ])
        record = load_jsonl_dataset(path)[0]
        assert record["id"] == "7"
        assert record["answer"] == "106"

    @pytest.mark.parametrize("missing", ["id", "problem", "answer"])
    def test_names_the_line_and_the_missing_key(self, tmp_path, missing):
        record = {"id": "a", "problem": "p", "answer": "1"}
        del record[missing]
        path = write_jsonl(tmp_path / "d.jsonl", [
            {"id": "ok", "problem": "p", "answer": "1"},
            record,
        ])
        with pytest.raises(ValueError, match=f"line 2.*{missing}"):
            load_jsonl_dataset(path)


class TestSelectSubset:
    problems = [{"id": str(i), "problem": "p", "answer": "1"} for i in range(100)]

    def test_returns_the_requested_size(self):
        assert len(select_subset(self.problems, 30, 123)) == 30

    def test_is_reproducible_for_a_given_seed(self):
        first = select_subset(self.problems, 30, 123)
        second = select_subset(self.problems, 30, 123)
        assert [p["id"] for p in first] == [p["id"] for p in second]

    def test_a_different_seed_picks_a_different_subset(self):
        a = [p["id"] for p in select_subset(self.problems, 30, 123)]
        b = [p["id"] for p in select_subset(self.problems, 30, 456)]
        assert a != b

    def test_keeps_the_original_dataset_order(self):
        picked = select_subset(self.problems, 30, 123)
        indices = [self.problems.index(p) for p in picked]
        assert indices == sorted(indices)

    def test_returns_everything_when_the_dataset_is_smaller_than_the_subset(self):
        assert select_subset(self.problems[:5], 30, 123) == self.problems[:5]

    def test_every_k_of_a_sweep_scores_the_same_problems(self):
        # main.py draws the subset once and reuses it, so this must not drift.
        runs = [select_subset(self.problems, 25, 123) for _ in range(5)]
        assert all([p["id"] for p in r] == [p["id"] for p in runs[0]] for r in runs)


class TestParseIntList:
    def test_parses_a_comma_separated_sweep(self):
        assert parse_int_list("1,10,100,1000") == [1, 10, 100, 1000]

    def test_tolerates_spaces_and_trailing_commas(self):
        assert parse_int_list("1, 10 ,100,") == [1, 10, 100]

    def test_a_single_value(self):
        assert parse_int_list("7") == [7]

    def test_empty(self):
        assert parse_int_list("") == []
