from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knowledge_agent.collectors.browser_session import BrowserSessionManager
from knowledge_agent.config import Settings
from knowledge_agent.storage.discovery_store import DiscoveryStore
from knowledge_agent.storage.raw_artifact_store import RawArtifactStore


@dataclass(frozen=True)
class ExtractedNote:
    note_id: str
    note_url: str
    title: str
    body_text: str
    image_urls: list[str]
    extracted_at: str


@dataclass(frozen=True)
class ExtractionResult:
    success: bool
    reason: str
    raw_path: str | None
    note: ExtractedNote | None


class NoteExtractor:
    def __init__(
        self,
        settings: Settings,
        discovery_store: DiscoveryStore,
        raw_store: RawArtifactStore,
    ) -> None:
        self._settings = settings
        self._discovery_store = discovery_store
        self._raw_store = raw_store

    def extract_note(self, note_id: str) -> ExtractionResult:
        discovered_note = self._discovery_store.get_note(note_id)
        if discovered_note is None:
            return ExtractionResult(
                success=False,
                reason=f"Unknown note_id: {note_id}",
                raw_path=None,
                note=None,
            )

        try:
            auth_result = BrowserSessionManager(self._settings).check_authentication()
            if not auth_result.is_authenticated:
                return ExtractionResult(
                    success=False,
                    reason=(
                        "Stored browser session is not ready for extraction: "
                        f"{auth_result.reason}"
                    ),
                    raw_path=None,
                    note=None,
                )

            with BrowserSessionManager(self._settings).open_context() as context:
                page = None
                try:
                    page = context.new_page()
                    page.goto(discovered_note.note_url, wait_until="domcontentloaded")
                    page.wait_for_timeout(3_000)
                    loaded_url = page.url
                    title = self._first_non_empty_text(
                        page,
                        [
                            "h1",
                            "meta[property='og:title']",
                            "title",
                        ],
                    )
                    body_text = self._safe_body_text(page)
                    image_urls = self._collect_image_urls(page)
                finally:
                    if page is not None:
                        page.close()
        except Exception as exc:
            return ExtractionResult(
                success=False,
                reason=f"Failed to extract note details: {exc}",
                raw_path=None,
                note=None,
            )

        if not self._looks_like_note_page(
            current_url=loaded_url,
            title=title,
            body_text=body_text,
            image_urls=image_urls,
        ):
            return ExtractionResult(
                success=False,
                reason="Loaded page does not look like a valid note detail page.",
                raw_path=None,
                note=None,
            )

        extracted_note = ExtractedNote(
            note_id=discovered_note.note_id,
            note_url=discovered_note.note_url,
            title=title,
            body_text=body_text,
            image_urls=image_urls,
            extracted_at=datetime.now(UTC).isoformat(),
        )
        raw_path = self._raw_store.save_raw_note(
            note_id=note_id,
            payload=asdict(extracted_note),
        )
        return ExtractionResult(
            success=True,
            reason="Extraction completed.",
            raw_path=str(raw_path),
            note=extracted_note,
        )

    def _looks_like_note_page(
        self,
        current_url: str,
        title: str,
        body_text: str,
        image_urls: list[str],
    ) -> bool:
        if not current_url.startswith(self._settings.xhs_base_url):
            return False
        if title:
            return True
        if len(body_text) >= 40:
            return True
        return bool(image_urls)

    def _first_non_empty_text(self, page: Any, selectors: list[str]) -> str:
        for selector in selectors:
            try:
                if selector.startswith("meta["):
                    value = page.locator(selector).first.get_attribute("content")
                elif selector == "title":
                    value = page.title()
                else:
                    value = page.locator(selector).first.inner_text(timeout=2_000)
            except Exception:
                continue
            if value is None:
                continue
            stripped_value = value.strip()
            if stripped_value:
                return stripped_value
        return ""

    def _safe_body_text(self, page: Any) -> str:
        try:
            return page.locator("body").inner_text(timeout=5_000).strip()
        except Exception:
            return ""

    def _collect_image_urls(self, page: Any) -> list[str]:
        try:
            image_urls = page.locator("img").evaluate_all(
                "elements => elements.map(element => element.src).filter(Boolean)"
            )
        except Exception:
            return []

        deduped_urls: list[str] = []
        seen_urls: set[str] = set()
        for image_url in image_urls:
            if not isinstance(image_url, str):
                continue
            if not image_url.startswith("http"):
                continue
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            deduped_urls.append(image_url)
        return deduped_urls
