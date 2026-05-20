from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from xhs_interview_answer_copilot.storage.project_deep_context_store import (
    ProjectDeepContextStore,
)
from xhs_interview_answer_copilot.workflows.schemas import ProjectDeepContext

_SKIP_DIRS = {
    ".git",
    ".memory",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "data",
    "outputs",
}

_TOPIC_KEYWORDS = {
    "retrieval": ["retriev", "search", "vector", "bm25", "hybrid", "embedding", "rag"],
    "memory": ["memory", "state", "artifact", "cache", "persist", "offset"],
    "worker": ["worker", "daemon", "tmux", "background", "poll", "loop"],
    "architecture": ["workflow", "graph", "router", "dispatch", "orchestration", "agent"],
    "storage": ["store", "sqlite", "db", "artifact", "vector"],
    "general": ["project", "repo", "system", "pipeline"],
}

_SENSITIVE_FILE_MARKERS = {
    ".env",
    "secret",
    "credential",
    "token",
    "passwd",
    "password",
    "private",
}

_SAFE_CONFIG_FILE_NAMES = {
    "package.json",
    "pyproject.toml",
    "prd.json",
}


class ProjectDeepScanWorkflow:
    _MAX_FINGERPRINT_BYTES = 131072
    _MAX_SCAN_FILE_BYTES = 262144

    def __init__(self, project_deep_context_store: ProjectDeepContextStore) -> None:
        self._project_deep_context_store = project_deep_context_store

    def run(
        self,
        *,
        project_path: str,
        topic: str,
        force_refresh: bool = False,
    ) -> tuple[bool, str, ProjectDeepContext | None, str | None]:
        root = Path(project_path).expanduser().resolve()
        if not root.exists():
            return False, f"Project path does not exist: {root}", None, None
        if not root.is_dir():
            return False, f"Project path is not a directory: {root}", None, None
        normalized_topic = self.normalize_topic(topic)
        fingerprint = self.project_fingerprint(root)
        if not force_refresh:
            cached = self._project_deep_context_store.load_context(
                project_path=str(root),
                topic=normalized_topic,
                fingerprint=fingerprint,
            )
            if cached is not None:
                return (
                    True,
                    "Loaded cached deep project context.",
                    cached,
                    str(
                        self._project_deep_context_store.context_path_for(
                            project_path=str(root),
                            topic=normalized_topic,
                        )
                    ),
                )
        try:
            context = self._build_context(root=root, topic=normalized_topic, fingerprint=fingerprint)
            context_path = self._project_deep_context_store.save_context(context)
        except Exception as exc:
            return False, f"Failed to deep scan project context: {exc}", None, None
        return True, "Deep project context refreshed.", context, str(context_path)

    @staticmethod
    def normalize_topic(topic: str) -> str:
        normalized = topic.strip().lower() or "general"
        return normalized if normalized in _TOPIC_KEYWORDS else "general"

    @staticmethod
    def project_fingerprint(project_root: Path) -> str:
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if dirty:
                tracked_changed = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD", "--", "."],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.splitlines()
                staged_changed = subprocess.run(
                    ["git", "diff", "--cached", "--name-only", "HEAD", "--", "."],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.splitlines()
                untracked_changed = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.splitlines()
                fingerprint_parts = [dirty]
                for relative_path in sorted({*tracked_changed, *staged_changed, *untracked_changed}):
                    candidate = project_root / relative_path
                    fingerprint_parts.append(relative_path)
                    if not candidate.exists() or not candidate.is_file():
                        continue
                    stat = candidate.stat()
                    fingerprint_parts.append(f"size={stat.st_size}:mtime={stat.st_mtime}")
                    with candidate.open("rb") as handle:
                        content = handle.read(ProjectDeepScanWorkflow._MAX_FINGERPRINT_BYTES)
                    fingerprint_parts.append(hashlib.sha1(content).hexdigest())
                fingerprint_source = "\n".join(fingerprint_parts)
                dirty_hash = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:12]
            else:
                dirty_hash = "clean"
            return f"git:{head}:{dirty_hash}"
        except Exception:
            latest_mtime = 0.0
            for candidate in project_root.rglob("*"):
                if not candidate.is_file() or ProjectDeepScanWorkflow._should_skip(candidate, project_root):
                    continue
                latest_mtime = max(latest_mtime, candidate.stat().st_mtime)
            digest = hashlib.sha1(f"{project_root}:{latest_mtime}".encode("utf-8")).hexdigest()[:16]
            return f"mtime:{digest}"

    def _build_context(self, *, root: Path, topic: str, fingerprint: str) -> ProjectDeepContext:
        scored_files = self._score_files(root=root, topic=topic)
        selected = scored_files[:6]
        if not selected:
            selected = [(root / "README.md", 1)] if (root / "README.md").exists() else []
        snippets: list[str] = []
        findings: list[str] = []
        key_files: list[str] = []
        for file_path, _ in selected:
            relative_path = str(file_path.relative_to(root))
            key_files.append(relative_path)
            extracted = self._extract_snippets(file_path=file_path, root=root, topic=topic)
            snippets.extend(extracted)
            findings.append(self._build_file_finding(relative_path, topic))
        summary = self._build_summary(project_name=root.name, topic=topic, key_files=key_files)
        return ProjectDeepContext(
            project_name=root.name,
            project_path=str(root),
            topic=topic,
            fingerprint=fingerprint,
            summary=summary,
            key_findings=self._dedupe(findings)[:8],
            key_files=key_files,
            code_snippets=snippets[:8],
        )

    def _score_files(self, *, root: Path, topic: str) -> list[tuple[Path, int]]:
        keywords = _TOPIC_KEYWORDS.get(topic, _TOPIC_KEYWORDS["general"])
        scored: list[tuple[Path, int]] = []
        for candidate in root.rglob("*"):
            if not candidate.is_file() or self._should_skip(candidate, root):
                continue
            if candidate.is_symlink():
                continue
            if not self._is_allowed_text_file(candidate):
                continue
            if self._looks_sensitive(candidate):
                continue
            if candidate.stat().st_size > self._MAX_SCAN_FILE_BYTES:
                continue
            relative_path = str(candidate.relative_to(root)).lower()
            score = sum(3 for keyword in keywords if keyword in relative_path)
            text = self._read_limited_text(candidate, limit=6000)
            lowered_text = text.lower()
            score += sum(1 for keyword in keywords if keyword in lowered_text)
            if score > 0:
                scored.append((candidate, score))
        scored.sort(key=lambda item: (-item[1], str(item[0])))
        return scored

    @staticmethod
    def _should_skip(candidate: Path, root: Path) -> bool:
        return any(part in _SKIP_DIRS for part in candidate.relative_to(root).parts)

    @staticmethod
    def _looks_sensitive(candidate: Path) -> bool:
        file_name = candidate.name.lower()
        if candidate.suffix.lower() in {".pem", ".key", ".crt"}:
            return True
        return any(marker in file_name for marker in _SENSITIVE_FILE_MARKERS)

    @staticmethod
    def _is_allowed_text_file(candidate: Path) -> bool:
        suffix = candidate.suffix.lower()
        if suffix in {".py", ".md", ".toml"}:
            return True
        if suffix in {".json", ".yaml", ".yml"}:
            return candidate.name.lower() in _SAFE_CONFIG_FILE_NAMES
        return False

    def _extract_snippets(self, *, file_path: Path, root: Path, topic: str) -> list[str]:
        keywords = _TOPIC_KEYWORDS.get(topic, _TOPIC_KEYWORDS["general"])
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        snippets: list[str] = []
        for index, line in enumerate(lines, start=1):
            lowered_line = line.lower()
            if not any(keyword in lowered_line for keyword in keywords):
                continue
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            start = max(index - 1, 1)
            end = min(index + 1, len(lines))
            block = []
            for line_no in range(start, end + 1):
                block.append(f"{line_no}: {lines[line_no - 1]}")
            snippets.append(
                f"{file_path.relative_to(root)}\n" + "\n".join(block)
            )
            if len(snippets) >= 2:
                break
        if snippets:
            return snippets
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "async def ")):
                start = index
                end = min(index + 2, len(lines))
                block = []
                for line_no in range(start, end + 1):
                    block.append(f"{line_no}: {lines[line_no - 1]}")
                snippets.append(f"{file_path.relative_to(root)}\n" + "\n".join(block))
                if len(snippets) >= 2:
                    break
        return snippets

    def _build_file_finding(self, relative_path: str, topic: str) -> str:
        return f"{relative_path} contains implementation details relevant to {topic}."

    def _build_summary(self, *, project_name: str, topic: str, key_files: list[str]) -> str:
        joined_files = ", ".join(key_files[:4]) if key_files else "repository metadata"
        return (
            f"Deep scan for the {project_name} project on topic={topic}. "
            f"Use {joined_files} as the primary implementation evidence for this topic-specific answer."
        )

    def _read_limited_text(self, path: Path, *, limit: int) -> str:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(limit)

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
