from __future__ import annotations

import base64
import importlib.resources
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.workflows.openai_clients import build_chat_model


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
        response = llm.invoke([message])
        content = getattr(response, "content", response)
        text = self._coerce_text(content)
        if text:
            return text
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
        return result.stdout.strip()

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
