from __future__ import annotations

import base64
import importlib.resources
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.workflows.llm_retry import is_budget_or_quota_error
from xhs_interview_answer_copilot.workflows.openai_clients import (
    build_chat_model,
    build_fallback_chat_model,
    fallback_available,
    fallback_model_name,
)


def _looks_like_low_quality_ocr(text: str) -> bool:
    compact = "".join(character for character in text if not character.isspace())
    if not compact:
        return True
    alpha_numeric_count = sum(character.isalnum() for character in compact)
    if alpha_numeric_count == 0:
        return True
    useful_ratio = alpha_numeric_count / len(compact)
    return len(compact) >= 12 and useful_ratio < 0.2


def _looks_like_invalid_model_ocr(text: str) -> bool:
    lowered = text.lower()
    suspicious_markers = [
        "last login:",
        "user@macbook",
        "% ls -la",
        "documents",
        "downloads",
        "desktop",
        ".ds_store",
    ]
    marker_hits = sum(marker in lowered for marker in suspicious_markers)
    return marker_hits >= 2


class MediaTextExtractor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def extract_from_paths(self, asset_paths: list[str]) -> tuple[list[str], list[str]]:
        extracted_texts: list[str] = []
        warnings: list[str] = []
        for asset_path in asset_paths:
            path = Path(asset_path)
            if not path.exists():
                warnings.append(f"Missing asset: {asset_path}")
                continue
            if not self._is_supported_image(path):
                warnings.append(f"Unsupported asset type: {asset_path}")
                continue
            try:
                extracted_text = self._extract_single_image(path)
            except Exception as exc:
                warnings.append(f"Failed to read {asset_path}: {exc}")
                continue
            if extracted_text:
                extracted_texts.append(extracted_text)
        return extracted_texts, warnings

    def _extract_single_image(self, path: Path) -> str:
        if self._settings.vision_model == self._settings.normalize_model:
            return self._extract_with_macos_vision(path)
        try:
            return self._extract_with_model(path)
        except Exception as exc:
            try:
                return self._extract_with_macos_vision(path)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Model OCR failed: {exc}; macOS Vision fallback failed: {fallback_exc}"
                )

    def _extract_with_model(self, path: Path) -> str:
        messages_module = __import__("langchain_core.messages", fromlist=["HumanMessage"])
        HumanMessage = messages_module.HumanMessage

        llm = build_chat_model(
            settings=self._settings,
            model_name=self._settings.vision_model,
            temperature=0,
        )
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{encoded}"
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Extract all visible text from this screenshot exactly. Keep line breaks where helpful. If there is no readable text, return an empty string.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"},
                },
            ]
        )
        try:
            response = llm.invoke([message])
        except Exception as exc:
            if not (fallback_available(self._settings) and is_budget_or_quota_error(exc)):
                raise
            fallback_llm = build_fallback_chat_model(
                settings=self._settings,
                model_name=fallback_model_name(self._settings, self._settings.vision_model),
                temperature=0,
            )
            response = fallback_llm.invoke([message])
        content = getattr(response, "content", response)
        text = self._coerce_text(content)
        if text and not _looks_like_invalid_model_ocr(text):
            return text
        if text:
            raise RuntimeError("Vision model returned invalid OCR text")
        raise RuntimeError("Vision model returned empty OCR text")

    def _extract_with_macos_vision(self, path: Path) -> str:
        tool_path = importlib.resources.files("xhs_interview_answer_copilot.tools").joinpath(
            "macos_vision_ocr.swift"
        )
        result = subprocess.run(
            ["swift", str(tool_path), str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        text = result.stdout.strip()
        if not text:
            raise RuntimeError("macOS Vision returned empty OCR text")
        if _looks_like_low_quality_ocr(text):
            raise RuntimeError("macOS Vision returned low-quality OCR text")
        return text

    def _is_supported_image(self, path: Path) -> bool:
        return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    def _coerce_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"].strip())
            return "\n".join(part for part in text_parts if part).strip()
        return str(content).strip()
