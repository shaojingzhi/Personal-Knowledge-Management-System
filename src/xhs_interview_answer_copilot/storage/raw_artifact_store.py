from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RawArtifactStore:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_note(self, note_id: str, payload: dict[str, Any]) -> Path:
        return self.save_raw_payload(note_id, payload)

    def save_raw_payload(self, bundle_id: str, payload: dict[str, Any]) -> Path:
        note_dir = self._output_dir / bundle_id
        note_dir.mkdir(parents=True, exist_ok=True)
        raw_path = note_dir / "raw.json"
        raw_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return raw_path

    def save_binary_asset(self, bundle_id: str, file_name: str, content: bytes) -> Path:
        bundle_dir = self._output_dir / bundle_id / "assets"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file_name).name or "attachment.bin"
        asset_path = bundle_dir / safe_name
        asset_path.write_bytes(content)
        return asset_path

    def load_raw_note(self, note_id: str) -> dict[str, Any] | None:
        return self.load_raw_payload(note_id)

    def load_raw_payload(self, bundle_id: str) -> dict[str, Any] | None:
        raw_path = self._output_dir / bundle_id / "raw.json"
        if not raw_path.exists():
            return None
        return json.loads(raw_path.read_text(encoding="utf-8"))
