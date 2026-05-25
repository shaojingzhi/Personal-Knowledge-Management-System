from __future__ import annotations

import hashlib
import math
import re
from importlib import import_module
from typing import Any, Literal

from xhs_interview_answer_copilot.config import Settings


def build_chat_model(
    settings: Settings,
    model_name: str,
    temperature: float,
    *,
    provider: Literal["primary", "fallback"] = "primary",
) -> Any:
    httpx = import_module("httpx")
    openai_module = import_module("langchain_openai")
    ChatOpenAI = openai_module.ChatOpenAI

    api_key, base_url, timeout_seconds = _resolve_provider_settings(settings=settings, provider=provider)
    client = _build_http_client(httpx=httpx, settings=settings, provider=provider)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout_seconds,
        max_retries=4,
        http_client=client,
    )


def build_embeddings_model(
    settings: Settings,
    *,
    provider: Literal["primary", "fallback"] = "primary",
) -> Any:
    if settings.embedding_model.startswith("local-hash"):
        return LocalHashEmbeddings()

    httpx = import_module("httpx")
    openai_module = import_module("langchain_openai")
    OpenAIEmbeddings = openai_module.OpenAIEmbeddings

    api_key, base_url, timeout_seconds = _resolve_provider_settings(settings=settings, provider=provider)
    client = _build_http_client(httpx=httpx, settings=settings, provider=provider)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        base_url=base_url,
        request_timeout=timeout_seconds,
        max_retries=4,
        http_client=client,
    )


def build_fallback_chat_model(settings: Settings, model_name: str, temperature: float) -> Any:
    return build_chat_model(settings=settings, model_name=model_name, temperature=temperature, provider="fallback")


def build_fallback_embeddings_model(settings: Settings) -> Any:
    return build_embeddings_model(settings=settings, provider="fallback")


def fallback_model_name(settings: Settings, model_name: str) -> str:
    return settings.fallback_model or model_name


def fallback_available(settings: Settings) -> bool:
    return settings.fallback_openai_api_key is not None


def _resolve_provider_settings(
    *,
    settings: Settings,
    provider: Literal["primary", "fallback"],
) -> tuple[str | None, str | None, int]:
    if provider == "fallback":
        return (
            settings.fallback_openai_api_key,
            settings.fallback_openai_base_url,
            settings.fallback_openai_timeout_seconds,
        )
    return settings.openai_api_key, settings.openai_base_url, settings.openai_timeout_seconds


def _build_http_client(httpx: Any, settings: Settings, provider: Literal["primary", "fallback"]) -> Any:
    if provider == "fallback":
        if settings.fallback_openai_proxy_url:
            return httpx.Client(proxy=settings.fallback_openai_proxy_url, timeout=settings.fallback_openai_timeout_seconds)
        return httpx.Client(timeout=settings.fallback_openai_timeout_seconds)
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
