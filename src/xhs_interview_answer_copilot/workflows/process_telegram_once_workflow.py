from __future__ import annotations

from importlib import import_module
from typing import NotRequired, TypedDict

from xhs_interview_answer_copilot.collectors.telegram_ingestor import TelegramIngestor
from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.dispatch.telegram_dispatcher import TelegramDispatcher
from xhs_interview_answer_copilot.storage.answer_artifact_store import AnswerArtifactStore
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
from xhs_interview_answer_copilot.storage.telegram_state_store import TelegramStateStore
from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
from xhs_interview_answer_copilot.workflows.generate_answers_workflow import GenerateAnswersWorkflow
from xhs_interview_answer_copilot.workflows.index_note_workflow import IndexNoteWorkflow
from xhs_interview_answer_copilot.workflows.normalize_note_workflow import NormalizeNoteWorkflow
from xhs_interview_answer_copilot.workflows.store_answer_records_workflow import (
    StoreAnswerRecordsWorkflow,
)


class BundleProcessState(TypedDict):
    bundle_id: str
    normalized_path: NotRequired[str]
    indexed_count: NotRequired[int]
    answer_path: NotRequired[str]
    markdown_path: NotRequired[str]
    stored_count: NotRequired[int]
    reply_message_id: NotRequired[int]
    reply_partial: NotRequired[bool]


class ProcessTelegramOnceWorkflow:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._raw_store = RawArtifactStore(output_dir=settings.output_dir)
        self._normalized_store = NormalizedArtifactStore(output_dir=settings.output_dir)
        self._answer_store = AnswerArtifactStore(output_dir=settings.output_dir)
        self._state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)
        self._vector_store = QuestionVectorStore(sqlite_path=settings.sqlite_path)

    def run_once(self) -> tuple[bool, str, list[str]]:
        previous_update_id = self._state_store.get_last_update_id()
        ingestor = TelegramIngestor(
            settings=self._settings,
            raw_store=self._raw_store,
            state_store=self._state_store,
        )
        ingest_result = ingestor.ingest_once()
        if "failed" in ingest_result.reason.lower() or "not configured" in ingest_result.reason.lower():
            return False, ingest_result.reason, []
        if ingest_result.saved_bundles == 0:
            return True, ingest_result.reason, []

        processed: list[str] = []
        for bundle_id in ingest_result.bundle_ids:
            success, reason = self._process_bundle(bundle_id)
            if not success:
                if "partial telegram reply" not in reason.lower():
                    self._restore_offset(previous_update_id, bundle_id)
                return False, f"{bundle_id}: {reason}", processed
            processed.append(bundle_id)
        return True, "Telegram processing completed.", processed

    def _restore_offset(self, previous_update_id: int | None, failed_bundle_id: str) -> None:
        failed_update_id = self._extract_update_id(failed_bundle_id)
        if failed_update_id is not None:
            self._state_store.set_last_update_id(max(failed_update_id - 1, 0))
            return
        if previous_update_id is None:
            self._state_store.set_last_update_id(0)
            return
        self._state_store.set_last_update_id(previous_update_id)

    def _extract_update_id(self, bundle_id: str) -> int | None:
        if not bundle_id.startswith("telegram_"):
            return None
        suffix = bundle_id.removeprefix("telegram_")
        return int(suffix) if suffix.isdigit() else None

    def _process_bundle(self, bundle_id: str) -> tuple[bool, str]:
        graph_module = import_module("langgraph.graph")
        END = graph_module.END
        START = graph_module.START
        StateGraph = graph_module.StateGraph

        graph = StateGraph(BundleProcessState)
        graph.add_node("normalize", self._normalize_node)
        graph.add_node("index", self._index_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("store_answers", self._store_answers_node)
        graph.add_node("reply", self._reply_node)
        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "index")
        graph.add_edge("index", "generate")
        graph.add_edge("generate", "store_answers")
        graph.add_edge("store_answers", "reply")
        graph.add_edge("reply", END)
        app = graph.compile()

        try:
            result = app.invoke({"bundle_id": bundle_id})
        except Exception as exc:
            return False, f"Automatic processing failed: {exc}"
        if result.get("reply_partial"):
            return False, "Partial Telegram reply sent."
        return True, "Processed bundle successfully."

    def _normalize_node(self, state: BundleProcessState) -> dict[str, object]:
        workflow = NormalizeNoteWorkflow(
            settings=self._settings,
            raw_store=self._raw_store,
            normalized_store=self._normalized_store,
        )
        success, reason, normalized_path = workflow.run(state["bundle_id"])
        if not success:
            raise RuntimeError(reason)
        return {"normalized_path": normalized_path or ""}

    def _index_node(self, state: BundleProcessState) -> dict[str, object]:
        workflow = IndexNoteWorkflow(
            settings=self._settings,
            normalized_store=self._normalized_store,
            vector_store=self._vector_store,
        )
        success, reason, indexed_count = workflow.run(state["bundle_id"])
        if not success:
            raise RuntimeError(reason)
        return {"indexed_count": indexed_count}

    def _generate_node(self, state: BundleProcessState) -> dict[str, object]:
        workflow = GenerateAnswersWorkflow(
            settings=self._settings,
            normalized_store=self._normalized_store,
            vector_store=self._vector_store,
            answer_store=self._answer_store,
        )
        success, reason, answer_path, markdown_path = workflow.run(state["bundle_id"])
        if not success:
            raise RuntimeError(reason)
        return {
            "answer_path": answer_path or "",
            "markdown_path": markdown_path or "",
        }

    def _store_answers_node(self, state: BundleProcessState) -> dict[str, object]:
        workflow = StoreAnswerRecordsWorkflow(
            settings=self._settings,
            answer_store=self._answer_store,
            vector_store=self._vector_store,
        )
        success, reason, stored_count = workflow.run(state["bundle_id"])
        if not success:
            raise RuntimeError(reason)
        return {"stored_count": stored_count}

    def _reply_node(self, state: BundleProcessState) -> dict[str, object]:
        dispatcher = TelegramDispatcher(
            settings=self._settings,
            answer_store=self._answer_store,
            raw_store=self._raw_store,
        )
        result = dispatcher.reply_answers(state["bundle_id"])
        if not result.success:
            if result.sent_count > 0:
                return {
                    "reply_message_id": result.message_id or 0,
                    "reply_partial": True,
                }
            raise RuntimeError(result.reason)
        return {"reply_message_id": result.message_id or 0}
