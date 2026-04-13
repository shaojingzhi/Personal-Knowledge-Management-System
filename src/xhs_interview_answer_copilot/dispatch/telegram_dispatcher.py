from __future__ import annotations

from dataclasses import dataclass

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
            return TelegramDispatchResult(False, f"Failed to load answer set: {exc}", None)
        if answer_set is None:
            return TelegramDispatchResult(False, f"Answer set not found for note_id: {note_id}", None)

        text = self._build_reply_text(answer_set)
        payload = {
            "chat_id": self._settings.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
        reply_to_message_id = self._read_reply_target(note_id)
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = str(reply_to_message_id)
        try:
            response = self._session.post(
                self._api_url("sendMessage"),
                data=payload,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                return TelegramDispatchResult(False, f"Telegram sendMessage failed: {payload}", None)
            message_id = payload.get("result", {}).get("message_id")
            return TelegramDispatchResult(True, "Telegram reply sent.", message_id)
        except Exception as exc:
            return TelegramDispatchResult(False, f"Telegram reply failed: {exc}", None)

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

    def _api_url(self, method: str) -> str:
        return f"{self._settings.telegram_api_base}/bot{self._settings.telegram_bot_token}/{method}"

    def _read_reply_target(self, note_id: str) -> int | None:
        raw_payload = self._raw_store.load_raw_payload(note_id)
        if raw_payload is None:
            return None
        metadata = raw_payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        message_id = metadata.get("message_id")
        return message_id if isinstance(message_id, int) else None
