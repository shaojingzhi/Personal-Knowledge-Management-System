from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xhs_interview_answer_copilot.workflows.schemas import ProjectContext


class ProjectContextStore:
    def __init__(self, output_dir: str) -> None:
        self._root_dir = Path(output_dir) / "project-context"
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _project_cache_dir(self, project_path: str) -> Path:
        normalized_path = str(Path(project_path).resolve())
        path_hash = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]
        safe_name = Path(normalized_path).name or "project"
        return self._root_dir / f"{safe_name}-{path_hash}"

    def load_context(self, project_path: str) -> ProjectContext | None:
        context_path = self._project_cache_dir(project_path) / "project_context.json"
        if not context_path.exists():
            return None
        try:
            return ProjectContext.model_validate_json(context_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_context(self, context: ProjectContext) -> Path:
        cache_dir = self._project_cache_dir(context.project_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        context_path = cache_dir / "project_context.json"
        context_path.write_text(
            json.dumps(context.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return context_path

    def context_path_for(self, project_path: str) -> Path:
        return self._project_cache_dir(project_path) / "project_context.json"
