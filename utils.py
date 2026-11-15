"""OpenAI API wrapper with a disk cache and retries.

Every answer is saved in .cache/ under a hash of the request, so running an
experiment again reads from disk instead of paying for the API calls.

Environment variables: OPENAI_REQUEST_TIMEOUT_S (default 120) and
OPENAI_MAX_ATTEMPTS (default 8).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

CACHE_DIR = Path(".cache")

# a 429 error message says how long to wait
RETRY_AFTER = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


class FatalModelError(RuntimeError):
    """Errors where retrying makes no sense (wrong key, wrong model, no quota)."""


@dataclass
class ModelOutput:
    text: str
    completion_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None


def _cache_path(request):
    key = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return CACHE_DIR / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def _load_cached(request):
    path = _cache_path(request)
    if not path.exists():
        return None
    return ModelOutput(**json.loads(path.read_text(encoding="utf-8")))


def _store_cached(request, output):
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(request)
    # write to a temp file first, so a crash can't leave a half-written cache entry
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(output), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _complete(client, request, max_attempts):
    kwargs = {
        "model": request["model"],
        "messages": [{"role": "user", "content": request["prompt"]}],
        "temperature": request["temperature"],
        "max_tokens": request["max_tokens"],
    }
    if request["seed"] is not None:
        kwargs["seed"] = request["seed"]

    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs)

        except AuthenticationError as err:
            raise FatalModelError("OPENAI_API_KEY was rejected (HTTP 401)") from err
        except PermissionDeniedError as err:
            raise FatalModelError(
                f"this key has no access to model '{request['model']}' (HTTP 403)"
            ) from err
        except BadRequestError as err:
            raise FatalModelError(f"OpenAI rejected the request (HTTP 400): {err}") from err

        except RateLimitError as err:
            # the per-minute limit resets by itself, the daily one doesn't
            if "requests per day" in str(err) or "RPD" in str(err):
                raise FatalModelError(f"daily request quota exhausted: {err}") from err
            attempt += 1
            if attempt >= max_attempts:
                raise
            match = RETRY_AFTER.search(str(err))
            time.sleep(float(match.group(1)) if match else 10.0)

        except (APITimeoutError, APIConnectionError):
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** attempt, 60))

        except APIStatusError as err:
            # only retry server errors (5xx), a 4xx would just fail again
            status = getattr(err, "status_code", None)
            if status is not None and status < 500:
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** attempt, 60))


def call_model(prompt, *, model="gpt-4o-mini", temperature=0.7, max_tokens=2048, seed=None):
    # The seed has to be part of the cache key. Otherwise all samples of a
    # problem would get the same cached answer.
    request = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "seed": seed,
    }

    cached = _load_cached(request)
    if cached is not None:
        return cached

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise FatalModelError("OPENAI_API_KEY is not set")

    client = OpenAI(
        api_key=api_key,
        timeout=float(os.environ.get("OPENAI_REQUEST_TIMEOUT_S", "120")),
        max_retries=0,  # retries are handled in _complete
    )
    response = _complete(client, request, int(os.environ.get("OPENAI_MAX_ATTEMPTS", "8")))

    usage = getattr(response, "usage", None)
    output = ModelOutput(
        text=response.choices[0].message.content or "",
        completion_tokens=getattr(usage, "completion_tokens", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
    _store_cached(request, output)
    return output
