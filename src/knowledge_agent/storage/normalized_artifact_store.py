from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_agent.workflows.schemas import NormalizedNote


class NormalizedArtifactStore:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_normalized_note(self, note_id: str, payload: dict[str, Any]) -> Path:
        note_dir = self._output_dir / note_id
        note_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = note_dir / "normalized.json"
        normalized_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized_path

    def load_normalized_note(self, note_id: str) -> NormalizedNote | None:
        normalized_path = self._output_dir / note_id / "normalized.json"
        if not normalized_path.exists():
            return None
        return NormalizedNote.model_validate_json(
            normalized_path.read_text(encoding="utf-8")
        )
