from __future__ import annotations

import re
from typing import Any


_LEADING_THINK_BLOCK_RE = re.compile(
    r"^\s*<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL
)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_pydantic_response(parser: Any, response: Any) -> Any:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return parser.parse(str(content))
    cleaned = _strip_reasoning_blocks(content)
    cleaned = _strip_code_fences(cleaned)
    return parser.parse(cleaned)


def _strip_reasoning_blocks(text: str) -> str:
    stripped = text
    while True:
        updated = _LEADING_THINK_BLOCK_RE.sub("", stripped, count=1)
        if updated == stripped:
            return stripped.strip()
        stripped = updated


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return _CODE_FENCE_RE.sub("", stripped).strip()
