"""Guarantees of the offline post-hoc rescoring in scripts/replay_corrected.py."""

import csv
import hashlib
import json
import socket
import subprocess
import sys

import pytest

import replay_corrected as rc

CACHE = rc.REPO_ROOT / ".cache"
OUT = rc.REPO_ROOT / "corrected_results"

needs_cache = pytest.mark.skipif(not CACHE.is_dir(), reason="response cache not available")
needs_outputs = pytest.mark.skipif(not (OUT / "manifest.json").is_file(), reason="run replay_corrected.py first")


def rows(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_original_csvs_are_unchanged_against_git():
    dirs = [cfg["original_dir"] for cfg in rc.RUNS.values()]
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *dirs], cwd=rc.REPO_ROOT)
    assert result.returncode == 0


@needs_outputs
def test_manifest_hashes_match_the_original_csvs():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["original_csv_sha256"].items():
        assert hashlib.sha256((rc.REPO_ROOT / name).read_bytes()).hexdigest() == digest


@needs_outputs
@pytest.mark.parametrize("run", list(rc.RUNS))
@pytest.mark.parametrize("k", rc.K_VALUES)
def test_ids_order_and_k_used_are_preserved(run, k):
    original = rows(rc.REPO_ROOT / rc.RUNS[run]["original_dir"] / f"k_{k}.csv")
    corrected = rows(OUT / run / f"k_{k}.csv")
    assert len(corrected) == 300
    assert len({r["example_id"] for r in corrected}) == 300
    assert [r["example_id"] for r in corrected] == [r["example_id"] for r in original]
    assert [r["k_used"] for r in corrected] == [r["k_used"] for r in original]
    assert [r["original_prediction"] for r in corrected] == [r["prediction"] for r in original]


@needs_outputs
def test_mcnemar_pairs_use_the_same_300_problems():
    tests = rows(OUT / "mcnemar.csv")
    assert len(tests) == 3 * 3 * 2
    assert all(t["problems"] == "300" for t in tests)


def test_rescoring_does_not_import_the_openai_client():
    code = ("import sys; sys.path[:0] = [r'%s', r'%s']; import replay_corrected; "
            "assert 'openai' not in sys.modules and 'utils' not in sys.modules"
            % (rc.REPO_ROOT, rc.REPO_ROOT / "scripts"))
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_network_is_disabled(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    rc.forbid_network()
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("api.openai.com", 443))


def test_cache_key_matches_utils():
    import utils

    prompt = rc.format_math_prompt("What is 1 + 1?")
    request = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 2048,
               "prompt": prompt, "seed": rc.derive_seed(3, 9)}
    assert rc.cache_file(utils.CACHE_DIR, "gpt-4o-mini", prompt, rc.derive_seed(3, 9)) == utils._cache_path(request)


def test_a_missing_cache_entry_fails_explicitly(tmp_path):
    with pytest.raises(rc.MissingCacheEntry, match="not in cache"):
        rc.rescore_run("gsm8k_4o", tmp_path)


@needs_cache
@needs_outputs
def test_every_cache_entry_used_is_verified():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["network_disabled"] and not manifest["openai_imported"]
    assert manifest["original_rows_reproduced"] == 12 * 300
    assert manifest["cache_entries_verified"] == sum(r["cache_entries_verified"] for r in manifest["runs"].values())
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest


@needs_cache
@needs_outputs
def test_rescoring_is_reproducible(tmp_path):
    subprocess.run([sys.executable, str(rc.REPO_ROOT / "scripts" / "replay_corrected.py"),
                    "--out-dir", str(tmp_path)], check=True, capture_output=True)
    fresh = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert fresh["output_sha256"] == saved["output_sha256"]
