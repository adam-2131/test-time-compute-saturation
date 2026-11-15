
import pytest

from analysis import (
    accuracy,
    discover_k_values,
    flips,
    is_correct,
    load_run,
    mcnemar_exact,
    shared_ids,
)

HEADER = "example_id,k_max,k_used,prediction,confidence,entropy,correct,runtime_seconds\n"


def write_run(directory, rows_by_k):
    directory.mkdir(parents=True, exist_ok=True)

    for k, rows in rows_by_k.items():
        lines = [HEADER]
        for example_id, correct in rows.items():
            lines.append(f"{example_id},{k},{k},7,0.5,1.0,{int(correct)},0.1\n")

        (directory / f"k_{k}.csv").write_text("".join(lines), encoding="utf-8")

    return directory


class TestMcnemarExact:
    def test_no_disagreements(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_equal_counts(self):
        assert mcnemar_exact(1, 1) == 1.0
        assert mcnemar_exact(10, 10) == 1.0

    @pytest.mark.parametrize(
        "lost, won, expected",
        [
            (0, 5, 2 / 32),
            (0, 10, 2 / 1024),
            (2, 1, 1.0),
            (7, 22, 0.008130),
        ],
    )
    def test_binomial_p_values(self, lost, won, expected):
        assert mcnemar_exact(lost, won) == pytest.approx(expected, abs=1e-6)

    def test_symmetric(self):
        for lost, won in [(0, 5), (3, 9), (7, 22)]:
            assert mcnemar_exact(lost, won) == mcnemar_exact(won, lost)

    def test_valid_p_value_range(self):
        for lost in range(8):
            for won in range(8):
                assert 0.0 <= mcnemar_exact(lost, won) <= 1.0

    def test_larger_difference(self):
        assert mcnemar_exact(5, 5) > mcnemar_exact(3, 7) > mcnemar_exact(0, 10)


class TestIsCorrect:
    def test_correct(self):
        assert is_correct({"correct": "1"}) is True

    def test_incorrect(self):
        assert is_correct({"correct": "0"}) is False


class TestDiscoverKValues:
    def test_sorted_values(self, tmp_path):
        run = write_run(tmp_path / "run", {
            10: {"a": 1},
            1: {"a": 1},
            100: {"a": 1},
        })
        assert discover_k_values(run) == [1, 10, 100]

    def test_other_k_values(self, tmp_path):
        run = write_run(tmp_path / "run", {
            3: {"a": 1},
            7: {"a": 1},
        })
        assert discover_k_values(run) == [3, 7]

    def test_ignores_other_files(self, tmp_path):
        run = write_run(tmp_path / "run", {1: {"a": 1}})

        (run / "notes.csv").write_text("x", encoding="utf-8")
        (run / "k_notanumber.csv").write_text("x", encoding="utf-8")

        assert discover_k_values(run) == [1]

    def test_empty_directory(self, tmp_path):
        (tmp_path / "empty").mkdir()

        with pytest.raises(FileNotFoundError):
            discover_k_values(tmp_path / "empty")


class TestSharedIds:
    def test_common_ids(self):
        a = {"x": {}, "y": {}, "z": {}}
        b = {"y": {}, "z": {}, "w": {}}

        assert shared_ids(a, b) == ["y", "z"]

    def test_order_preserved(self):
        a = {"z": {}, "y": {}, "x": {}}
        b = {"x": {}, "y": {}, "z": {}}

        assert shared_ids(a, b) == ["z", "y", "x"]

    def test_single_run(self):
        assert shared_ids({"b": {}, "a": {}}) == ["b", "a"]


class TestLoadRun:
    def test_shared_examples(self, tmp_path):
        run = write_run(tmp_path / "run", {
            1: {"a": 1, "b": 0, "c": 0},
            10: {"a": 1, "b": 1},
        })

        by_k, shared = load_run(run)

        assert sorted(by_k) == [1, 10]
        assert shared == ["a", "b"]

    def test_accuracy_on_shared_ids(self, tmp_path):
        run = write_run(tmp_path / "run", {
            1: {"a": 1, "b": 0, "c": 1},
            10: {"a": 1, "b": 1},
        })

        by_k, shared = load_run(run)

        assert accuracy(by_k[1], shared) == 0.5
        assert accuracy(by_k[10], shared) == 1.0


class TestFlips:
    def test_flip_directions(self, tmp_path):
        run = write_run(tmp_path / "run", {
            1: {"a": 1, "b": 0, "c": 1, "d": 0},
            10: {"a": 0, "b": 1, "c": 1, "d": 0},
        })

        by_k, shared = load_run(run)
        lost, won = flips(by_k[1], by_k[10], shared)

        assert lost == ["a"]
        assert won == ["b"]

    def test_no_flips(self, tmp_path):
        run = write_run(tmp_path / "run", {
            1: {"a": 1, "b": 0},
            10: {"a": 1, "b": 0},
        })

        by_k, shared = load_run(run)

        assert flips(by_k[1], by_k[10], shared) == ([], [])
