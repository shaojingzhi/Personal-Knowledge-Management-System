from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

from knowledge_agent.collectors.browser_session import BrowserSessionManager
from knowledge_agent.config import Settings
from knowledge_agent.storage.discovery_store import DiscoveredNote, DiscoveryStore

NOTE_ID_PATTERN = re.compile(r"/(?:explore|discovery/item)/([a-zA-Z0-9]+)")


@dataclass(frozen=True)
class DiscoveryResult:
    discovered_notes: list[DiscoveredNote]
    scanned_links: int
    reason: str


class FavoritesCollector:
    def __init__(self, settings: Settings, store: DiscoveryStore) -> None:
        self._settings = settings
        self._store = store

    def discover_new_notes(self) -> DiscoveryResult:
        if self._settings.xhs_favorites_url is None:
            return DiscoveryResult([], 0, "XHS_FAVORITES_URL is not configured.")

        auth_result = BrowserSessionManager(self._settings).check_authentication()
        if not auth_result.is_authenticated:
            return DiscoveryResult([], 0, auth_result.reason)

        try:
            with BrowserSessionManager(self._settings).open_context() as context:
                page = None
                try:
                    page = context.new_page()
                    page.goto(self._settings.xhs_favorites_url, wait_until="domcontentloaded")
                    page.wait_for_timeout(3_000)
                    hrefs = page.locator(self._settings.xhs_note_link_selector).evaluate_all(
                        "elements => elements.map(element => element.href).filter(Boolean)"
                    )
                finally:
                    if page is not None:
                        page.close()
        except Exception as exc:
            return DiscoveryResult([], 0, f"Failed to load favorites page: {exc}")

        new_notes: list[DiscoveredNote] = []
        seen_note_ids: set[str] = set()
        scanned_links = 0
        discovered_at = datetime.now(UTC).isoformat()
        for href in hrefs:
            note_url = urljoin(self._settings.xhs_base_url, href)
            note_id = self._extract_note_id(note_url)
            if note_id is None:
                continue
            scanned_links += 1
            if note_id in seen_note_ids or self._store.has_note(note_id):
                continue
            seen_note_ids.add(note_id)
            new_notes.append(
                DiscoveredNote(
                    note_id=note_id,
                    note_url=note_url,
                    discovered_at=discovered_at,
                )
            )

        self._store.save_notes(new_notes)
        return DiscoveryResult(
            discovered_notes=new_notes,
            scanned_links=scanned_links,
            reason="Discovery completed.",
        )

    def _extract_note_id(self, url: str) -> str | None:
        match = NOTE_ID_PATTERN.search(url)
        if match is None:
            return None
        return match.group(1)
