from __future__ import annotations

import sqlite3
from pathlib import Path


class TelegramStateStore:
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
                CREATE TABLE IF NOT EXISTS bot_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL
                )
                """
            )

    def get_last_update_id(self) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_value FROM bot_state WHERE state_key = 'telegram_last_update_id'"
            ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def set_last_update_id(self, update_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO bot_state (state_key, state_value)
                VALUES ('telegram_last_update_id', ?)
                """,
                (str(update_id),),
            )
