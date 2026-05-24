from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xhs_interview_answer_copilot.workflows.schemas import ProjectDeepContext


class ProjectDeepContextStore:
    def __init__(self, output_dir: str) -> None:
        self._root_dir = Path(output_dir) / "project-context"
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _project_cache_dir(self, project_path: str) -> Path:
        normalized_path = str(Path(project_path).resolve())
        path_hash = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]
        safe_name = Path(normalized_path).name or "project"
        return self._root_dir / f"{safe_name}-{path_hash}"

    def load_context(
        self,
        *,
        project_path: str,
        topic: str,
        fingerprint: str,
        provider: str,
        question_hash: str | None,
    ) -> ProjectDeepContext | None:
        context_path = self._topic_cache_path(
            project_path=project_path,
            topic=topic,
            provider=provider,
            question_hash=question_hash,
        )
        if not context_path.exists():
            return None
        try:
            context = ProjectDeepContext.model_validate_json(context_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return context if context.fingerprint == fingerprint else None

    def save_context(
        self,
        context: ProjectDeepContext,
        *,
        provider: str,
        question_hash: str | None,
    ) -> Path:
        topic_path = self._topic_cache_path(
            project_path=context.project_path,
            topic=context.topic,
            provider=provider,
            question_hash=question_hash,
        )
        topic_path.parent.mkdir(parents=True, exist_ok=True)
        topic_path.write_text(
            json.dumps(context.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return topic_path

    def context_path_for(
        self,
        *,
        project_path: str,
        topic: str,
        provider: str,
        question_hash: str | None,
    ) -> Path:
        return self._topic_cache_path(
            project_path=project_path,
            topic=topic,
            provider=provider,
            question_hash=question_hash,
        )

    def _topic_cache_path(
        self,
        *,
        project_path: str,
        topic: str,
        provider: str,
        question_hash: str | None,
    ) -> Path:
        safe_topic = topic.replace("/", "-").replace(" ", "-")
        suffix = f"_{question_hash}" if question_hash else ""
        return self._project_cache_dir(project_path) / f"deep_{provider}_{safe_topic}{suffix}.json"
