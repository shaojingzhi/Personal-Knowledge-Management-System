from __future__ import annotations

import json
import tomllib
from pathlib import Path

from knowledge_agent.storage.project_context_store import ProjectContextStore
from knowledge_agent.workflows.schemas import ProjectContext

_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "data",
    "outputs",
}


class ProjectContextWorkflow:
    def __init__(self, project_context_store: ProjectContextStore) -> None:
        self._project_context_store = project_context_store

    def run(
        self,
        project_path: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[bool, str, ProjectContext | None, str | None]:
        root = Path(project_path).expanduser().resolve()
        if not root.exists():
            return False, f"Project path does not exist: {root}", None, None
        if not root.is_dir():
            return False, f"Project path is not a directory: {root}", None, None
        if not force_refresh:
            cached = self._project_context_store.load_context(str(root))
            if cached is not None:
                return (
                    True,
                    "Loaded cached project context.",
                    cached,
                    str(self._project_context_store.context_path_for(str(root))),
                )
        try:
            context = self._build_context(root)
            context_path = self._project_context_store.save_context(context)
        except Exception as exc:
            return False, f"Failed to build project context: {exc}", None, None
        return True, "Project context refreshed.", context, str(context_path)

    def _build_context(self, root: Path) -> ProjectContext:
        metadata = self._read_project_metadata(root)
        key_files = self._collect_key_files(root)
        return ProjectContext(
            project_name=self._project_name(root, metadata),
            project_path=str(root),
            summary=self._build_summary(root, metadata),
            tech_stack=self._build_tech_stack(metadata),
            memory_system=self._build_memory_system(key_files),
            retrieval_system=self._build_retrieval_system(key_files),
            workflow_orchestration=self._build_workflow_orchestration(key_files),
            storage=self._build_storage_summary(key_files),
            background_jobs=self._build_background_jobs_summary(key_files),
            key_files=key_files,
        )

    def _read_project_metadata(self, root: Path) -> dict[str, object]:
        metadata: dict[str, object] = {}
        pyproject_path = root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                metadata["pyproject"] = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        package_json_path = root / "package.json"
        if package_json_path.exists():
            try:
                metadata["package_json"] = json.loads(package_json_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        readme_path = root / "README.md"
        if readme_path.exists():
            try:
                metadata["readme"] = self._read_limited_text(readme_path)
            except Exception:
                pass
        prd_path = root / "docs" / "PRD.md"
        if prd_path.exists():
            try:
                metadata["prd"] = self._read_limited_text(prd_path)
            except Exception:
                pass
        return metadata

    def _project_name(self, root: Path, metadata: dict[str, object]) -> str:
        pyproject = metadata.get("pyproject")
        if isinstance(pyproject, dict):
            project = pyproject.get("project")
            if isinstance(project, dict):
                name = project.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
        package_json = metadata.get("package_json")
        if isinstance(package_json, dict):
            name = package_json.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return root.name

    def _build_summary(self, root: Path, metadata: dict[str, object]) -> str:
        readme = metadata.get("readme")
        if isinstance(readme, str) and readme.strip():
            first_meaningful_line = self._first_descriptive_line(readme)
            if first_meaningful_line:
                return (
                    f"{root.name} is the active project. {first_meaningful_line} "
                    "Use the key files below as the primary evidence when answering project-specific interview questions."
                )
        prd = metadata.get("prd")
        if isinstance(prd, str) and prd.strip():
            first_meaningful_line = self._first_descriptive_line(prd)
            if first_meaningful_line:
                return (
                    f"{root.name} is the active project. {first_meaningful_line} "
                    "The project context is derived from planning docs and implementation entrypoints."
                )
        return (
            f"{root.name} is the active project. The context summary is derived from repository metadata, "
            "entrypoints, workflow files, and storage modules."
        )

    def _build_tech_stack(self, metadata: dict[str, object]) -> list[str]:
        stack: list[str] = []
        pyproject = metadata.get("pyproject")
        if isinstance(pyproject, dict):
            project = pyproject.get("project")
            if isinstance(project, dict):
                stack.append("Python")
                dependencies = project.get("dependencies")
                if isinstance(dependencies, list):
                    for dependency in dependencies:
                        if not isinstance(dependency, str):
                            continue
                        lowered = dependency.lower()
                        if "langgraph" in lowered:
                            stack.append("LangGraph")
                        elif "langchain" in lowered:
                            stack.append("LangChain")
                        elif "pydantic" in lowered:
                            stack.append("Pydantic")
                        elif "playwright" in lowered:
                            stack.append("Playwright")
                        elif "requests" in lowered or "httpx" in lowered:
                            stack.append("HTTP API clients")
        package_json = metadata.get("package_json")
        if isinstance(package_json, dict):
            stack.append("Node.js")
        return self._dedupe(stack)

    def _build_memory_system(self, key_files: list[str]) -> str:
        if any("memory" in path.lower() for path in key_files):
            return (
                "The project includes explicit memory-related modules, so answers should describe those files first "
                "and then explain how that memory state is persisted or reused."
            )
        if any("telegram_state_store" in path for path in key_files) or any(
            "artifact_store" in path for path in key_files
        ):
            return (
                "The project does not expose a dedicated autonomous memory subsystem in the scanned files. "
                "Instead, durable state is kept through lightweight stores such as bot state, vector records, "
                "and local JSON or Markdown artifacts that can be reused across runs."
            )
        return (
            "No strong memory-specific implementation was detected from the bounded scan. "
            "Answer conservatively and point to durable state or cached artifacts only when supported by key files."
        )

    def _build_retrieval_system(self, key_files: list[str]) -> str:
        if any("retrieve" in path.lower() for path in key_files) and any(
            "vector_store" in path.lower() for path in key_files
        ):
            return (
                "Retrieval is implemented through a dedicated retrieval workflow plus a vector store layer. "
                "If bm25 or hybrid files are present, explain ranking fusion and mode switching rather than only embeddings."
            )
        if any("vector_store" in path.lower() for path in key_files):
            return (
                "Retrieval appears to rely on indexed local question or answer records stored in a vector-style persistence layer."
            )
        return "No dedicated retrieval implementation was detected in the scanned key files."

    def _build_workflow_orchestration(self, key_files: list[str]) -> str:
        if any("process_telegram_once_workflow" in path for path in key_files) or any(
            "workflow" in path.lower() for path in key_files
        ):
            return (
                "The project uses workflow modules as its orchestration backbone. "
                "When answering architecture questions, describe how source ingestion, normalization, retrieval, generation, and reply stages are sequenced."
            )
        if any(path.endswith("main.py") for path in key_files):
            return "The main entrypoint coordinates the primary command flow and dispatches into focused modules."
        return "No strong orchestration file was detected in the bounded scan."

    def _build_storage_summary(self, key_files: list[str]) -> str:
        storage_files = [path for path in key_files if "store" in path.lower() or "storage/" in path]
        if storage_files:
            return (
                "Persistent state is handled through focused storage modules rather than one monolithic database layer. "
                "Use the key storage files as evidence for how artifacts, offsets, and indexed records are saved."
            )
        return "No dedicated storage module was detected in the bounded scan."

    def _build_background_jobs_summary(self, key_files: list[str]) -> str:
        if any("daemon" in path.lower() for path in key_files) or any(
            "worker" in path.lower() for path in key_files
        ):
            return (
                "The project includes resident worker or daemon-style background execution. "
                "When asked, explain how long-running processing is separated from quick user-visible replies."
            )
        return "No dedicated background worker was detected in the scanned key files."

    def _collect_key_files(self, root: Path) -> list[str]:
        explicit_candidates = [
            "README.md",
            "docs/PRD.md",
            "pyproject.toml",
            "package.json",
            "src/knowledge_agent/main.py",
            "src/knowledge_agent/config.py",
            "src/knowledge_agent/workflows/process_telegram_once_workflow.py",
            "src/knowledge_agent/workflows/process_telegram_daemon_workflow.py",
            "src/knowledge_agent/workflows/generate_answers_workflow.py",
            "src/knowledge_agent/workflows/normalize_note_workflow.py",
            "src/knowledge_agent/workflows/retrieve_questions.py",
            "src/knowledge_agent/storage/telegram_state_store.py",
            "src/knowledge_agent/storage/vector_store.py",
            "src/knowledge_agent/storage/answer_artifact_store.py",
            "src/knowledge_agent/dispatch/telegram_dispatcher.py",
        ]
        selected: list[str] = []
        seen: set[str] = set()
        for relative_path in explicit_candidates:
            candidate = root / relative_path
            if candidate.exists() and candidate.is_file():
                selected.append(relative_path)
                seen.add(relative_path)

        fallback_patterns = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "src/**/*retriev*.py",
            "src/**/*vector*.py",
            "src/**/*store*.py",
            "src/**/*workflow*.py",
            "src/**/*telegram*.py",
            "src/**/*memory*.py",
            "src/**/*.py",
            "app/**/*.py",
            "lib/**/*.py",
        ]
        for pattern in fallback_patterns:
            for candidate in root.glob(pattern):
                if not candidate.is_file() or self._should_skip(candidate, root):
                    continue
                relative_path = str(candidate.relative_to(root))
                if relative_path in seen:
                    continue
                selected.append(relative_path)
                seen.add(relative_path)
                if len(selected) >= 15:
                    return selected
        return selected

    def _should_skip(self, candidate: Path, root: Path) -> bool:
        return any(part in _SKIP_DIRS for part in candidate.relative_to(root).parts)

    def _first_descriptive_line(self, text: str) -> str:
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            line = stripped
            if line:
                return line
        return ""

    def _read_limited_text(self, path: Path, limit: int = 6000) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
