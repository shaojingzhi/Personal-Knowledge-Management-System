from __future__ import annotations

import hashlib
import math
import re
from importlib import import_module
from typing import Any

from xhs_interview_answer_copilot.config import Settings


def build_chat_model(settings: Settings, model_name: str, temperature: float) -> Any:
    httpx = import_module("httpx")
    openai_module = import_module("langchain_openai")
    ChatOpenAI = openai_module.ChatOpenAI

    client = _build_http_client(httpx=httpx, settings=settings)
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
        timeout=settings.openai_timeout_seconds,
        http_client=client,
    )


def build_embeddings_model(settings: Settings) -> Any:
    if settings.embedding_model.startswith("local-hash"):
        return LocalHashEmbeddings()

    httpx = import_module("httpx")
    openai_module = import_module("langchain_openai")
    OpenAIEmbeddings = openai_module.OpenAIEmbeddings

    client = _build_http_client(httpx=httpx, settings=settings)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        request_timeout=settings.openai_timeout_seconds,
        http_client=client,
    )


def _build_http_client(httpx: Any, settings: Settings) -> Any:
    if settings.openai_proxy_url:
        return httpx.Client(proxy=settings.openai_proxy_url, timeout=settings.openai_timeout_seconds)
    return httpx.Client(timeout=settings.openai_timeout_seconds)


class LocalHashEmbeddings:
    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % self._dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
