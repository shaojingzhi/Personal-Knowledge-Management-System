from __future__ import annotations

import time
from typing import Any


def invoke_with_retry(
    chain: Any,
    payload: dict[str, object],
    *,
    max_attempts: int = 4,
    base_delay_seconds: float = 2.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1 or not _is_retryable_rate_limit(exc):
                raise
            time.sleep(base_delay_seconds * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM invocation failed without an explicit exception")


def _is_retryable_rate_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "1302" in message


def is_budget_or_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "budget has been exceeded" in message
        or "budget_exceeded" in message
        or "quota" in message
        or "insufficient_quota" in message
        or "余额" in message
        or "额度" in message
    )
