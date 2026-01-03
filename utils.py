from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import time
from typing import Any, TypeVar

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel

@dataclass
class ModelOutput:
    text: str
    completion_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None
    raw: object  # keep raw response for debugging

TModel = TypeVar("TModel", bound=BaseModel)


def call_model(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: int | None = None,
    cache_salt: int | None = None,
) -> ModelOutput:
    """
    Single model call that returns:
    - generated text
    - token usage (needed for test-time compute accounting)
    
    Args:
        seed: Optional seed for deterministic sampling (passed to OpenAI API)
        cache_salt: Optional value included ONLY in the cache key. Useful for making
            repeated samples with identical prompts independent without changing the
            model-visible prompt/messages.
    """
    os.makedirs(".cache", exist_ok=True)
    cache_key_payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "seed": seed,
    }
    if cache_salt is not None:
        # Important: do NOT include this in the prompt/messages; it is cache-only.
        # Also, omit entirely when None to preserve legacy cache keys.
        cache_key_payload["cache_salt"] = cache_salt
    cache_key_str = json.dumps(
        cache_key_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()
    cache_path = os.path.join(".cache", f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return ModelOutput(
            text=cached.get("text", ""),
            completion_tokens=cached.get("completion_tokens", None),
            prompt_tokens=cached.get("prompt_tokens", None),
            total_tokens=cached.get("total_tokens", None),
            raw=None,
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    attempts = 0
    max_attempts = 5
    while True:
        try:
            api_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                api_kwargs["seed"] = seed
            resp = client.chat.completions.create(**api_kwargs)
            break
        except (APITimeoutError, APIConnectionError) as e:
            attempts += 1
            if attempts >= max_attempts:
                raise
            wait_s = min(2 ** attempts, 60)  # exponential backoff, max 60s
            time.sleep(wait_s)
        except RateLimitError as e:
            attempts += 1
            msg = str(e)
            if "requests per day" in msg or "RPD" in msg:
                raise
            if attempts >= max_attempts:
                raise
            m = re.search(r"try again in ([0-9.]+)s", str(e), flags=re.IGNORECASE)
            wait_s = float(m.group(1)) if m else 10.0
            time.sleep(wait_s)

    text = resp.choices[0].message.content or ""

    usage = getattr(resp, "usage", None)
    completion = getattr(usage, "completion_tokens", None) if usage else None
    prompt_toks = getattr(usage, "prompt_tokens", None) if usage else None
    total = getattr(usage, "total_tokens", None) if usage else None

    tmp_cache_path = cache_path + ".tmp"
    with open(tmp_cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": text,
                "completion_tokens": completion,
                "prompt_tokens": prompt_toks,
                "total_tokens": total,
            },
            f,
            ensure_ascii=False,
        )
    os.replace(tmp_cache_path, cache_path)

    return ModelOutput(
        text=text,
        completion_tokens=completion,
        prompt_tokens=prompt_toks,
        total_tokens=total,
        raw=resp,
    )


def _strip_json_fences(text: str) -> str:
    """
    Removes common markdown fences like:
      ```json
      {...}
      ```
    """
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_json_value(text: str) -> Any:
    """
    Best-effort JSON extraction for model outputs.
    Tries full-string parse, then falls back to extracting the outermost {...} or [...].
    """
    s = _strip_json_fences(text)
    if not s:
        raise ValueError("empty response")

    # Try strict parse first.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the largest JSON object/array region.
    first_obj = s.find("{")
    last_obj = s.rfind("}")
    first_arr = s.find("[")
    last_arr = s.rfind("]")

    candidates: list[str] = []
    if 0 <= first_obj < last_obj:
        candidates.append(s[first_obj : last_obj + 1])
    if 0 <= first_arr < last_arr:
        candidates.append(s[first_arr : last_arr + 1])

    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue

    raise ValueError("failed to parse JSON from response")


def _schema_hint(model_cls: type[BaseModel]) -> str:
    """
    Returns a JSON schema string for both Pydantic v2 and v1.
    """
    if hasattr(model_cls, "model_json_schema"):  # pydantic v2
        schema_obj = model_cls.model_json_schema()
        return json.dumps(schema_obj, ensure_ascii=False, indent=2)
    # pydantic v1 fallback
    schema_obj = model_cls.schema()  # type: ignore[attr-defined]
    return json.dumps(schema_obj, ensure_ascii=False, indent=2)


def _validate_with_pydantic(model_cls: type[TModel], data: Any) -> TModel:
    if hasattr(model_cls, "model_validate"):  # pydantic v2
        return model_cls.model_validate(data)  # type: ignore[no-any-return]
    return model_cls.parse_obj(data)  # type: ignore[attr-defined,no-any-return]


def sample(
    prompt: str,
    model_cls: type[TModel],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_attempts: int = 3,
) -> TModel:
    """
    Sample a *JSON-only* response and validate it against a Pydantic model.
    This is intentionally minimal (no tool calling) to keep the repo lightweight.
    """
    schema = _schema_hint(model_cls)
    instruction = (
        "Return ONLY valid JSON (no markdown, no code fences, no commentary). "
        "The JSON MUST validate against this schema:\n"
        f"{schema}"
    )
    full_prompt = f"{prompt}\n\n{instruction}\n"

    last_err: Exception | None = None
    for _ in range(max_attempts):
        try:
            out = call_model(
                full_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            data = _extract_json_value(out.text)
            return _validate_with_pydantic(model_cls, data)
        except Exception as e:  # noqa: BLE001 - surface a clean error after retries
            last_err = e
            continue

    raise RuntimeError(f"Failed to sample valid JSON for {model_cls.__name__}: {last_err}")