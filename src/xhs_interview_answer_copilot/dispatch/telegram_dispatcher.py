from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.answer_artifact_store import AnswerArtifactStore
from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
from xhs_interview_answer_copilot.workflows.schemas import GeneratedAnswerSet


@dataclass(frozen=True)
class TelegramDispatchResult:
    success: bool
    reason: str
    message_id: int | None
    sent_count: int = 0


class TelegramDispatcher:
    def __init__(
        self,
        settings: Settings,
        answer_store: AnswerArtifactStore,
        raw_store: RawArtifactStore,
    ) -> None:
        self._settings = settings
        self._answer_store = answer_store
        self._raw_store = raw_store
        self._session = requests.Session()
        if self._settings.telegram_proxy_url is not None:
            self._session.proxies.update(
                {
                    "http": self._settings.telegram_proxy_url,
                    "https": self._settings.telegram_proxy_url,
                }
            )

    def reply_answers(self, note_id: str) -> TelegramDispatchResult:
        if self._settings.telegram_bot_token is None or self._settings.telegram_chat_id is None:
            return TelegramDispatchResult(False, "Telegram bot token or chat id is not configured.", None)

        try:
            answer_set = self._answer_store.load_answers(note_id)
        except Exception as exc:
            return TelegramDispatchResult(False, f"Failed to load answer set: {exc}", None, 0)
        if answer_set is None:
            return TelegramDispatchResult(False, f"Answer set not found for note_id: {note_id}", None, 0)

        markdown_text = self._answer_store.load_markdown(note_id)
        text = self._build_full_reply_text(markdown_text) if markdown_text else self._build_reply_text(answer_set)
        chunks = self._split_text(text)
        reply_to_message_id = self._read_reply_target(note_id)
        first_message_id: int | None = None
        sent_count = 0
        try:
            current_reply_target = reply_to_message_id
            for chunk in chunks:
                payload = {
                    "chat_id": self._settings.telegram_chat_id,
                    "text": chunk,
                    "disable_web_page_preview": "true",
                }
                if current_reply_target is not None:
                    payload["reply_to_message_id"] = str(current_reply_target)
                response = self._session.post(
                    self._api_url("sendMessage"),
                    data=payload,
                    timeout=60,
                )
                response.raise_for_status()
                response_payload = response.json()
                if not response_payload.get("ok"):
                    return TelegramDispatchResult(
                        False,
                        f"Telegram sendMessage failed: {response_payload}",
                        first_message_id,
                        sent_count,
                    )
                sent_count += 1
                message_id = response_payload.get("result", {}).get("message_id")
                if first_message_id is None and isinstance(message_id, int):
                    first_message_id = message_id
                current_reply_target = message_id if isinstance(message_id, int) else current_reply_target
            return TelegramDispatchResult(True, "Telegram reply sent.", first_message_id, sent_count)
        except Exception as exc:
            return TelegramDispatchResult(
                False,
                f"Telegram reply failed: {exc}",
                first_message_id,
                sent_count,
            )

    def send_status_message(self, note_id: str, text: str) -> TelegramDispatchResult:
        return self._send_message(
            text=text,
            reply_to_message_id=self._read_reply_target(note_id),
            timeout_seconds=10,
        )

    def send_message_to_chat(self, text: str) -> TelegramDispatchResult:
        return self._send_message(text=text, timeout_seconds=10)

    def send_answer_markdown_document(self, note_id: str, caption: str) -> TelegramDispatchResult:
        if self._settings.telegram_bot_token is None or self._settings.telegram_chat_id is None:
            return TelegramDispatchResult(False, "Telegram bot token or chat id is not configured.", None)
        markdown_path = self._answer_store.get_answer_markdown_path(note_id)
        if not markdown_path.exists():
            return TelegramDispatchResult(False, f"Markdown answer not found for note_id: {note_id}", None)
        last_error = ""
        for _ in range(3):
            try:
                with markdown_path.open("rb") as file:
                    data = {
                        "chat_id": self._settings.telegram_chat_id,
                        "caption": caption[:1024],
                    }
                    reply_to_message_id = self._read_reply_target(note_id)
                    if reply_to_message_id is not None:
                        data["reply_to_message_id"] = str(reply_to_message_id)
                    response = self._session.post(
                        self._api_url("sendDocument"),
                        data=data,
                        files={"document": (Path(markdown_path).name, file, "text/markdown")},
                        timeout=120,
                    )
                response.raise_for_status()
                response_payload = response.json()
                if not response_payload.get("ok"):
                    last_error = f"Telegram sendDocument failed: {response_payload}"
                    continue
                message_id = response_payload.get("result", {}).get("message_id")
                return TelegramDispatchResult(
                    True,
                    "Telegram markdown document sent.",
                    message_id if isinstance(message_id, int) else None,
                    1,
                )
            except Exception as exc:
                last_error = str(exc)
        return TelegramDispatchResult(False, f"Telegram markdown document failed: {last_error}", None, 0)

    def _send_message(
        self,
        text: str,
        reply_to_message_id: int | None = None,
        timeout_seconds: int = 60,
    ) -> TelegramDispatchResult:
        if self._settings.telegram_bot_token is None or self._settings.telegram_chat_id is None:
            return TelegramDispatchResult(False, "Telegram bot token or chat id is not configured.", None)
        try:
            payload = {
                "chat_id": self._settings.telegram_chat_id,
                "text": text[:3900],
                "disable_web_page_preview": "true",
            }
            if reply_to_message_id is not None:
                payload["reply_to_message_id"] = str(reply_to_message_id)
            response = self._session.post(
                self._api_url("sendMessage"),
                data=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
            if not response_payload.get("ok"):
                return TelegramDispatchResult(False, f"Telegram sendMessage failed: {response_payload}", None, 0)
            message_id = response_payload.get("result", {}).get("message_id")
            return TelegramDispatchResult(
                True,
                "Telegram message sent.",
                message_id if isinstance(message_id, int) else None,
                1,
            )
        except Exception as exc:
            return TelegramDispatchResult(False, f"Telegram message failed: {exc}", None, 0)

    def _build_reply_text(self, answer_set: GeneratedAnswerSet) -> str:
        lines = [
            f"{answer_set.title or answer_set.note_id}",
            f"题目数：{len(answer_set.answers)}",
            "",
        ]
        for index, answer in enumerate(answer_set.answers[:5], start=1):
            lines.extend(
                [
                    f"Q{index}. {answer.question}",
                    answer.short_answer,
                    "",
                ]
            )
        text = "\n".join(lines).strip()
        return text[:3500]

    def _build_full_reply_text(self, markdown_text: str) -> str:
        filtered_lines: list[str] = []
        skip_next_blank_after_grounding = False
        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- Source ID:") or stripped.startswith("- Source URL:"):
                continue
            if stripped == "**Grounding Source IDs**":
                skip_next_blank_after_grounding = True
                continue
            if skip_next_blank_after_grounding:
                if not stripped:
                    continue
                if stripped.startswith("`"):
                    continue
                skip_next_blank_after_grounding = False
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    def _split_text(self, text: str, limit: int = 3200) -> list[str]:
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        current = ""
        for block in text.split("\n\n"):
            candidate = block if not current else f"{current}\n\n{block}"
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(block) <= limit:
                current = block
                continue
            start = 0
            while start < len(block):
                chunks.append(block[start : start + limit])
                start += limit
            current = ""
        if current:
            chunks.append(current)
        return chunks

    def _api_url(self, method: str) -> str:
        return f"{self._settings.telegram_api_base}/bot{self._settings.telegram_bot_token}/{method}"

    def _read_reply_target(self, note_id: str) -> int | None:
        raw_payload = self._raw_store.load_raw_payload(note_id)
        if raw_payload is None:
            return None
        metadata = raw_payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        reply_target_message_id = metadata.get("reply_target_message_id")
        if isinstance(reply_target_message_id, int):
            return reply_target_message_id
        message_id = metadata.get("message_id")
        return message_id if isinstance(message_id, int) else None
