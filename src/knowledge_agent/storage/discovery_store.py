from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiscoveredNote:
    note_id: str
    note_url: str
    discovered_at: str


class DiscoveryStore:
    def __init__(self, sqlite_path: str) -> None:
        self._sqlite_path = Path(sqlite_path)
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._sqlite_path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS discovered_notes (
                    note_id TEXT PRIMARY KEY,
                    note_url TEXT NOT NULL,
                    discovered_at TEXT NOT NULL
                )
                """
            )

    def has_note(self, note_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM discovered_notes WHERE note_id = ? LIMIT 1",
                (note_id,),
            ).fetchone()
        return row is not None

    def get_note(self, note_id: str) -> DiscoveredNote | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT note_id, note_url, discovered_at
                FROM discovered_notes
                WHERE note_id = ?
                LIMIT 1
                """,
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        return DiscoveredNote(
            note_id=row[0],
            note_url=row[1],
            discovered_at=row[2],
        )

    def save_notes(self, notes: list[DiscoveredNote]) -> int:
        if not notes:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO discovered_notes (note_id, note_url, discovered_at)
                VALUES (?, ?, ?)
                """,
                [(note.note_id, note.note_url, note.discovered_at) for note in notes],
            )
            return connection.total_changes
