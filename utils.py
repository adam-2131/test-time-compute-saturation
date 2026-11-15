
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
RETRY_AFTER = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


class FatalModelError(RuntimeError):
    pass


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

    for attempt in range(max(1, max_attempts)):
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

        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as err:
            if isinstance(err, RateLimitError):
                if "requests per day" in str(err) or "RPD" in str(err):
                    raise FatalModelError(f"daily request quota exhausted: {err}") from err

                match = RETRY_AFTER.search(str(err))
                delay = float(match.group(1)) if match else 10.0

            else:
                if isinstance(err, APIStatusError):
                    status = getattr(err, "status_code", None)
                    if status is not None and status < 500:
                        raise

                delay = min(2 ** (attempt + 1), 60)

            if attempt + 1 >= max_attempts:
                raise

            time.sleep(delay)


def call_model(
    prompt, *, model="gpt-4o-mini", temperature=0.7,
    max_tokens=2048, seed=None
):
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
        max_retries=0,
    )

    response = _complete(
        client,
        request,
        int(os.environ.get("OPENAI_MAX_ATTEMPTS", "8")),
    )

    usage = getattr(response, "usage", None)

    output = ModelOutput(
        text=response.choices[0].message.content or "",
        completion_tokens=getattr(usage, "completion_tokens", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )

    _store_cached(request, output)
    return output
