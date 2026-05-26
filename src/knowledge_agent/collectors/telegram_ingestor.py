from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from knowledge_agent.config import Settings
from knowledge_agent.storage.raw_artifact_store import RawArtifactStore
from knowledge_agent.storage.telegram_state_store import TelegramStateStore
from knowledge_agent.workflows.schemas import SourceBundle, SourceLink


@dataclass(frozen=True)
class TelegramIngestResult:
    processed_updates: int
    saved_bundles: int
    bundle_ids: list[str]
    reason: str


class TelegramIngestor:
    def __init__(
        self,
        settings: Settings,
        raw_store: RawArtifactStore,
        state_store: TelegramStateStore,
    ) -> None:
        self._settings = settings
        self._raw_store = raw_store
        self._state_store = state_store
        self._session = requests.Session()
        if self._settings.telegram_proxy_url is not None:
            self._session.proxies.update(
                {
                    "http": self._settings.telegram_proxy_url,
                    "https": self._settings.telegram_proxy_url,
                }
            )

    def ingest_once(self, commit_offset: bool = True) -> TelegramIngestResult:
        if self._settings.telegram_bot_token is None:
            return TelegramIngestResult(0, 0, [], "TELEGRAM_BOT_TOKEN is not configured.")

        last_update_id = self._state_store.get_last_update_id()
        offset = None if last_update_id is None else last_update_id + 1
        try:
            updates = self._get_updates(offset=offset)
        except Exception as exc:
            return TelegramIngestResult(0, 0, [], f"Telegram ingestion failed: {exc}")

        processed_updates = 0
        saved_bundles = 0
        bundle_ids: list[str] = []
        max_update_id = last_update_id
        for update in updates:
            processed_updates += 1
            update_id = int(update["update_id"])
            max_update_id = update_id if max_update_id is None else max(max_update_id, update_id)
            message = update.get("message") or update.get("channel_post")
            if message is None:
                continue
            if not self._is_allowed_message(message):
                continue
            bundle_id = f"telegram_{update_id}"
            try:
                asset_paths = self._download_assets(bundle_id=bundle_id, message=message)
            except Exception:
                asset_paths = []
            bundle = self._build_source_bundle(
                bundle_id=bundle_id,
                update_id=update_id,
                message=message,
                asset_paths=asset_paths,
            )
            self._raw_store.save_raw_payload(
                bundle_id=bundle_id,
                payload=bundle.model_dump(mode="json"),
            )
            saved_bundles += 1
            bundle_ids.append(bundle_id)

        if commit_offset and max_update_id is not None:
            self._state_store.set_last_update_id(max_update_id)
        return TelegramIngestResult(
            processed_updates,
            saved_bundles,
            bundle_ids,
            "Telegram ingestion completed.",
        )

    def _get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        response = self._session.get(
            self._api_url("getUpdates"),
            params={"offset": offset, "timeout": 1} if offset is not None else {"timeout": 1},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {payload}")
        return list(payload.get("result", []))

    def _download_assets(self, bundle_id: str, message: dict[str, Any]) -> list[str]:
        asset_paths: list[str] = []
        photo_list = message.get("photo", [])
        if photo_list:
            largest_photo = photo_list[-1]
            downloaded = self._download_file(bundle_id, largest_photo["file_id"])
            if downloaded is not None:
                asset_paths.append(downloaded)

        document = message.get("document")
        if document is not None:
            downloaded = self._download_file(bundle_id, document["file_id"], document.get("file_name"))
            if downloaded is not None:
                asset_paths.append(downloaded)
        return asset_paths

    def _download_file(
        self,
        bundle_id: str,
        file_id: str,
        file_name: str | None = None,
    ) -> str | None:
        file_response = self._session.get(
            self._api_url("getFile"),
            params={"file_id": file_id},
            timeout=30,
        )
        file_response.raise_for_status()
        payload = file_response.json()
        if not payload.get("ok"):
            return None
        file_path = payload["result"]["file_path"]
        download_url = (
            f"{self._settings.telegram_api_base}/file/bot{self._settings.telegram_bot_token}/{file_path}"
        )
        content_response = self._session.get(download_url, timeout=60)
        content_response.raise_for_status()
        resolved_name = file_name or Path(file_path).name
        saved_path = self._raw_store.save_binary_asset(
            bundle_id=bundle_id,
            file_name=resolved_name,
            content=content_response.content,
        )
        return str(saved_path)

    def _is_allowed_message(self, message: dict[str, Any]) -> bool:
        if self._settings.telegram_chat_id is None:
            return True
        chat = message.get("chat", {})
        return str(chat.get("id")) == self._settings.telegram_chat_id

    def _api_url(self, method: str) -> str:
        return f"{self._settings.telegram_api_base}/bot{self._settings.telegram_bot_token}/{method}"

    def _build_source_bundle(
        self,
        bundle_id: str,
        update_id: int,
        message: dict[str, Any],
        asset_paths: list[str],
    ) -> SourceBundle:
        text = self._read_text(message.get("text"))
        caption = self._read_text(message.get("caption"))
        text_blocks = [value for value in [text, caption] if value]
        links = self._extract_links(
            text=text,
            caption=caption,
            entities=message.get("entities", []),
            caption_entities=message.get("caption_entities", []),
        )
        return SourceBundle(
            bundle_id=bundle_id,
            source="telegram",
            source_type="message",
            canonical_url=links[0].url if links else "",
            title=self._build_title(text_blocks),
            text_blocks=text_blocks,
            links=links,
            asset_paths=asset_paths,
            image_urls=[],
            metadata={
                "update_id": update_id,
                "message_id": message.get("message_id"),
                "chat": message.get("chat"),
                "date": message.get("date"),
                "photo_count": len(message.get("photo", [])),
                "document": message.get("document"),
                "entities": message.get("entities", []),
                "caption_entities": message.get("caption_entities", []),
                "raw_message": message,
            },
        )

    def _read_text(self, value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _build_title(self, text_blocks: list[str]) -> str:
        if not text_blocks:
            return ""
        return text_blocks[0][:80]

    def _extract_links(
        self,
        text: str,
        caption: str,
        entities: object,
        caption_entities: object,
    ) -> list[SourceLink]:
        links: list[SourceLink] = []
        seen_urls: set[str] = set()
        for entity_list, source_text in ((entities, text), (caption_entities, caption)):
            if not isinstance(entity_list, list):
                continue
            for entity in entity_list:
                if not isinstance(entity, dict):
                    continue
                url = entity.get("url") if entity.get("type") == "text_link" else None
                if isinstance(url, str) and url not in seen_urls:
                    seen_urls.add(url)
                    links.append(SourceLink(url=url, label=""))
                    continue
                if entity.get("type") != "url":
                    continue
                offset = entity.get("offset")
                length = entity.get("length")
                if not isinstance(offset, int) or not isinstance(length, int):
                    continue
                extracted = source_text[offset : offset + length].strip()
                if extracted and extracted not in seen_urls:
                    seen_urls.add(extracted)
                    links.append(SourceLink(url=extracted, label=""))
        return links
