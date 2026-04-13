from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AnswerArtifactStore:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_answers(self, note_id: str, payload: dict[str, Any]) -> Path:
        note_dir = self._output_dir / note_id
        note_dir.mkdir(parents=True, exist_ok=True)
        answer_path = note_dir / "answers.json"
        answer_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return answer_path
