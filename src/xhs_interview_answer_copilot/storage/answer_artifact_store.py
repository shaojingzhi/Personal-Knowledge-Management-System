from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xhs_interview_answer_copilot.workflows.schemas import GeneratedAnswerSet


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

    def save_markdown(self, note_id: str, markdown: str) -> Path:
        note_dir = self._output_dir / note_id
        note_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = note_dir / "answer.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        return markdown_path

    def load_answers(self, note_id: str) -> GeneratedAnswerSet | None:
        answer_path = self.get_answer_json_path(note_id)
        if not answer_path.exists():
            return None
        return GeneratedAnswerSet.model_validate_json(answer_path.read_text(encoding="utf-8"))

    def get_answer_json_path(self, note_id: str) -> Path:
        return self._output_dir / note_id / "answers.json"

    def get_answer_markdown_path(self, note_id: str) -> Path:
        return self._output_dir / note_id / "answer.md"
