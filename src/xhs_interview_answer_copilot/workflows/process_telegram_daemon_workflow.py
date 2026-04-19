from __future__ import annotations

import time
from dataclasses import dataclass

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.workflows.process_telegram_once_workflow import (
    ProcessTelegramOnceWorkflow,
)


@dataclass(frozen=True)
class DaemonRunResult:
    success: bool
    reason: str
    loops: int
    processed_bundles: int


class ProcessTelegramDaemonWorkflow:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(
        self,
        *,
        interval_seconds: int,
        failure_backoff_seconds: int,
        max_loops: int | None = None,
    ) -> DaemonRunResult:
        if interval_seconds <= 0:
            return DaemonRunResult(False, "interval_seconds must be greater than 0", 0, 0)
        if failure_backoff_seconds <= 0:
            return DaemonRunResult(False, "failure_backoff_seconds must be greater than 0", 0, 0)

        workflow = ProcessTelegramOnceWorkflow(settings=self._settings)
        loops = 0
        processed_bundles = 0

        try:
            while max_loops is None or loops < max_loops:
                try:
                    success, reason, processed = workflow.run_once()
                except Exception as exc:
                    success, reason, processed = False, f"Daemon loop failed: {exc}", []
                loops += 1
                processed_bundles += len(processed)
                self._log_iteration(loops=loops, success=success, reason=reason, processed=processed)
                if self._should_stop_after_failure(success=success, reason=reason):
                    return DaemonRunResult(False, reason, loops, processed_bundles)
                sleep_seconds = interval_seconds if success else failure_backoff_seconds
                if max_loops is not None and loops >= max_loops:
                    break
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            return DaemonRunResult(
                True,
                "Telegram daemon stopped by user.",
                loops,
                processed_bundles,
            )

        return DaemonRunResult(
            True,
            "Telegram daemon completed requested loop count.",
            loops,
            processed_bundles,
        )

    def _log_iteration(
        self,
        *,
        loops: int,
        success: bool,
        reason: str,
        processed: list[str],
    ) -> None:
        status = "ok" if success else "error"
        print(
            f"[loop {loops}] status={status} processed={len(processed)} reason={reason} bundles={processed}"
        )

    def _should_stop_after_failure(self, *, success: bool, reason: str) -> bool:
        if success:
            return False
        return "Partial Telegram reply sent." in reason
