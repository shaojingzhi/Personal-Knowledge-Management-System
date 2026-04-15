from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
from xhs_interview_answer_copilot.workflows.schemas import SourceBundle, SourceLink


@dataclass(frozen=True)
class FeishuIngestResult:
    success: bool
    reason: str
    bundle_id: str | None = None
    challenge: str | None = None


class FeishuIngestor:
    def __init__(self, raw_store: RawArtifactStore) -> None:
        self._raw_store = raw_store

    def ingest_event_file(self, file_path: str) -> FeishuIngestResult:
        try:
            payload_text = Path(file_path).read_text(encoding="utf-8")
        except Exception as exc:
            return FeishuIngestResult(False, f"Failed to read Feishu event file: {exc}")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return FeishuIngestResult(False, f"Invalid Feishu event JSON: {exc}")
        return self.ingest_event_payload(payload)

    def ingest_event_payload(self, payload: dict[str, Any]) -> FeishuIngestResult:
        if payload.get("type") == "url_verification":
            challenge = payload.get("challenge")
            if isinstance(challenge, str):
                return FeishuIngestResult(True, "Feishu url verification event parsed.", None, challenge)
            return FeishuIngestResult(False, "Feishu url verification payload missing challenge.")

        header = payload.get("header")
        event = payload.get("event")
        if not isinstance(header, dict) or not isinstance(event, dict):
            return FeishuIngestResult(False, "Unsupported Feishu payload shape.")
        if header.get("event_type") != "im.message.receive_v1":
            return FeishuIngestResult(False, f"Unsupported Feishu event type: {header.get('event_type')}")

        message = event.get("message")
        sender = event.get("sender")
        if not isinstance(message, dict) or not isinstance(sender, dict):
            return FeishuIngestResult(False, "Feishu event missing message or sender.")
        if message.get("message_type") != "text":
            return FeishuIngestResult(False, f"Unsupported Feishu message type: {message.get('message_type')}")

        content_text = self._extract_text_content(message.get("content"))
        if not content_text:
            return FeishuIngestResult(False, "Feishu text message is empty.")

        message_id = message.get("message_id")
        event_id = header.get("event_id")
        if not isinstance(message_id, str) or not isinstance(event_id, str):
            return FeishuIngestResult(False, "Feishu payload missing stable ids.")

        bundle_id = f"feishu_{self._sanitize_id(message_id)}"
        bundle = SourceBundle(
            bundle_id=bundle_id,
            source="feishu",
            source_type="message",
            canonical_url=self._extract_first_url(content_text),
            title=content_text.splitlines()[0][:80] if content_text else "",
            text_blocks=[content_text],
            links=[SourceLink(url=url, label="") for url in self._extract_all_urls(content_text)],
            asset_paths=[],
            image_urls=[],
            metadata={
                "event_id": event_id,
                "event_type": header.get("event_type"),
                "create_time": header.get("create_time"),
                "message_id": message_id,
                "chat_id": message.get("chat_id"),
                "chat_type": message.get("chat_type"),
                "sender": sender,
                "message_preview": content_text[:200],
            },
        )
        self._raw_store.save_raw_payload(bundle_id=bundle_id, payload=bundle.model_dump(mode="json"))
        return FeishuIngestResult(True, "Feishu event ingested.", bundle_id=bundle_id)

    def _extract_text_content(self, raw_content: object) -> str:
        if not isinstance(raw_content, str):
            return ""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            return raw_content.strip()
        text = parsed.get("text")
        return text.strip() if isinstance(text, str) else ""

    def _extract_first_url(self, text: str) -> str:
        urls = self._extract_all_urls(text)
        return urls[0] if urls else ""

    def _extract_all_urls(self, text: str) -> list[str]:
        urls: list[str] = []
        for token in text.split():
            if token.startswith("http://") or token.startswith("https://"):
                cleaned = token.rstrip(").,!?]}>\"'"
                )
                if cleaned not in urls:
                    urls.append(cleaned)
        return urls

    def _sanitize_id(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", value)
        return sanitized[:120] or "event"
