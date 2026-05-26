from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_agent.workflows.schemas import ProjectAnswerMemoryRecord


class ProjectAnswerMemoryStore:
    def __init__(self, output_dir: str) -> None:
        self._root_dir = Path(output_dir) / "project-context"
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def append_record(self, record: ProjectAnswerMemoryRecord) -> Path:
        memory_path = self._memory_path(record.project_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
        return memory_path

    def load_recent_records(
        self,
        *,
        project_path: str,
        project_fingerprint: str | None = None,
        topic: str | None = None,
        limit: int = 5,
    ) -> list[ProjectAnswerMemoryRecord]:
        memory_path = self._memory_path(project_path)
        if not memory_path.exists():
            return []
        records: list[ProjectAnswerMemoryRecord] = []
        for raw_line in reversed(memory_path.read_text(encoding="utf-8").splitlines()):
            if not raw_line.strip():
                continue
            try:
                record = ProjectAnswerMemoryRecord.model_validate_json(raw_line)
            except Exception:
                continue
            if project_fingerprint is not None and record.project_fingerprint != project_fingerprint:
                continue
            if topic is not None and record.topic != topic:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        records.reverse()
        return records

    def memory_path_for(self, project_path: str) -> Path:
        return self._memory_path(project_path)

    def _memory_path(self, project_path: str) -> Path:
        normalized_path = str(Path(project_path).resolve())
        path_hash = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]
        safe_name = Path(normalized_path).name or "project"
        return self._root_dir / f"{safe_name}-{path_hash}" / "answer_memory.jsonl"
