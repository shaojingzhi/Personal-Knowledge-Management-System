from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class TraceEvent:
    event_id: str
    step_name: str
    started_at: str
    finished_at: str
    duration_ms: int
    status: str
    summary: str
    input_preview: str = ""
    output_preview: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "step_name": self.step_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "summary": self.summary,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentTrace:
    trace_id: str
    workflow_name: str
    question: str
    started_at: str
    finished_at: str = ""
    status: str = "running"
    final_summary: str = ""
    events: list[TraceEvent] = field(default_factory=list)

    @classmethod
    def start(cls, workflow_name: str, question: str) -> "AgentTrace":
        return cls(
            trace_id=uuid.uuid4().hex[:12],
            workflow_name=workflow_name,
            question=question,
            started_at=_now_iso(),
        )

    def add_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def finish(self, *, status: str, final_summary: str) -> None:
        self.finished_at = _now_iso()
        self.status = status
        self.final_summary = _safe_text(final_summary, limit=200)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "workflow_name": self.workflow_name,
            "question": _safe_text(self.question, limit=160),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "final_summary": self.final_summary,
            "events": [event.to_dict() for event in self.events],
        }


class TraceRecorder:
    def __init__(self, output_dir: str) -> None:
        self._root_dir = Path(output_dir) / "traces"

    def save(self, trace: AgentTrace) -> Path:
        self._root_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = self._root_dir / trace.workflow_name / trace.trace_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.json"
        trace_path.write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return trace_path


def observable_step(
    step_name: str,
    *,
    summarize_input: Callable[..., str] | None = None,
    summarize_output: Callable[[Any], str] | None = None,
    summarize_error: Callable[[Exception], str] | None = None,
    build_summary: Callable[[Any], str] | None = None,
    metadata_builder: Callable[[Any], dict[str, Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            owner = args[0]
            trace = getattr(owner, "_active_trace", None)
            started_at = _now_iso()
            started_perf = time.perf_counter()
            input_preview = ""
            if summarize_input is not None:
                try:
                    input_preview = _safe_text(summarize_input(*args, **kwargs))
                except Exception:
                    input_preview = ""
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                finished_at = _now_iso()
                duration_ms = int((time.perf_counter() - started_perf) * 1000)
                if trace is not None:
                    trace.add_event(
                        TraceEvent(
                            event_id=uuid.uuid4().hex[:12],
                            step_name=step_name,
                            started_at=started_at,
                            finished_at=finished_at,
                            duration_ms=duration_ms,
                            status="error",
                            summary=(
                                _safe_text(summarize_error(exc))
                                if summarize_error is not None
                                else _safe_text(str(exc), limit=200)
                            ),
                            input_preview=input_preview,
                            error=_safe_text(str(exc), limit=200),
                        )
                    )
                raise
            finished_at = _now_iso()
            duration_ms = int((time.perf_counter() - started_perf) * 1000)
            if trace is not None:
                output_preview = ""
                summary = "step completed"
                metadata: dict[str, Any] = {}
                if summarize_output is not None:
                    try:
                        output_preview = _safe_text(summarize_output(result))
                    except Exception:
                        output_preview = ""
                if build_summary is not None:
                    try:
                        summary = _safe_text(build_summary(result))
                    except Exception:
                        summary = "step completed"
                if metadata_builder is not None:
                    try:
                        metadata = metadata_builder(result)
                    except Exception:
                        metadata = {}
                trace.add_event(
                    TraceEvent(
                        event_id=uuid.uuid4().hex[:12],
                        step_name=step_name,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=duration_ms,
                        status="ok",
                        summary=summary,
                        input_preview=input_preview,
                        output_preview=output_preview,
                        metadata=metadata,
                    )
                )
            return result

        return wrapper

    return decorator


def render_cli_timeline(trace: AgentTrace) -> str:
    lines = [f"trace_id={trace.trace_id}", "trace_timeline="]
    for index, event in enumerate(trace.events, start=1):
        lines.append(
            f"  {index}. [{event.status}] {event.step_name} {event.duration_ms}ms - {event.summary}"
        )
    if trace.final_summary:
        lines.append(f"trace_summary={trace.final_summary}")
    return "\n".join(lines)


def _safe_text(value: str, limit: int = 400) -> str:
    compact = " ".join(str(value).split())
    compact = _redact_sensitive_text(compact)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    patterns = [
        r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)",
        r"(?i)(token\s*[=:]\s*)([^\s,;]+)",
        r"(?i)(password\s*[=:]\s*)([^\s,;]+)",
        r"(?i)(secret\s*[=:]\s*)([^\s,;]+)",
        r"(?i)(bearer\s+)([^\s,;]+)",
        r"\bsk-[A-Za-z0-9_-]{10,}\b",
    ]
    for pattern in patterns:
        redacted = re.sub(pattern, _replace_match, redacted)
    return redacted


def _replace_match(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}[REDACTED]"
    return "[REDACTED]"
