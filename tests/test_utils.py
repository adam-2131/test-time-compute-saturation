"""Response cache and finish_reason handling, without contacting the API."""

import json
from types import SimpleNamespace

import pytest

import utils


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "CACHE_DIR", tmp_path)
    return tmp_path


def request(**changes):
    req = {"model": "m", "temperature": 0.7, "max_tokens": 2048, "prompt": "p", "seed": 1}
    req.update(changes)
    return req


def fake_response(text, finish_reason):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=7, prompt_tokens=3, total_tokens=10),
    )


def test_old_cache_entries_without_finish_reason_still_load(cache):
    old = {"text": "Final Answer: 1", "completion_tokens": 5, "prompt_tokens": 2, "total_tokens": 7}
    utils._cache_path(request()).write_text(json.dumps(old), encoding="utf-8")

    output = utils.call_model("p", model="m", seed=1)

    assert output.text == "Final Answer: 1"
    assert output.finish_reason is None


def test_max_tokens_is_part_of_the_cache_key():
    assert utils._cache_path(request()) != utils._cache_path(request(max_tokens=16384))


def test_new_responses_store_finish_reason(cache, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(utils, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(utils, "_complete", lambda *args: fake_response("cut", "length"))

    output = utils.call_model("p", model="m", max_tokens=16384, seed=1)
    stored = json.loads(utils._cache_path(request(max_tokens=16384)).read_text(encoding="utf-8"))

    assert output.finish_reason == "length"
    assert stored["finish_reason"] == "length"


def test_cached_responses_do_not_call_the_api(cache, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(utils, "OpenAI", lambda **kwargs: object())
    calls = []
    monkeypatch.setattr(utils, "_complete", lambda *args: calls.append(1) or fake_response("x", "stop"))

    utils.call_model("p", model="m", seed=1)
    utils.call_model("p", model="m", seed=1)

    assert len(calls) == 1


def test_missing_key_is_fatal_for_uncached_requests(cache, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(utils.FatalModelError):
        utils.call_model("p", model="m", seed=1)
