from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
import subprocess
import sys
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
from xhs_interview_answer_copilot.workflows.schemas import SourceBundle


class BundleProcessState(TypedDict):
    bundle_id: str
    normalized_path: NotRequired[str]
    indexed_count: NotRequired[int]
    answer_path: NotRequired[str]
    markdown_path: NotRequired[str]
    stored_count: NotRequired[int]
    reply_message_id: NotRequired[int]
    reply_partial: NotRequired[bool]
    backfill_started: NotRequired[bool]


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
        ingest_result = ingestor.ingest_once(commit_offset=False)
        if "failed" in ingest_result.reason.lower() or "not configured" in ingest_result.reason.lower():
            return False, ingest_result.reason, []
        if ingest_result.saved_bundles == 0:
            return True, ingest_result.reason, []

        grouped_bundle_ids = self._group_bundle_ids(ingest_result.bundle_ids)
        processed: list[str] = []
        for bundle_id in grouped_bundle_ids:
            success, reason = self._process_bundle(bundle_id)
            if not success:
                if not processed:
                    self._restore_offset(previous_update_id, bundle_id)
                return False, f"{bundle_id}: {reason}", processed
            self._commit_processed_bundle(bundle_id)
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
        source_bundle = self._load_source_bundle(bundle_id)
        if source_bundle is not None:
            metadata = source_bundle.metadata
            child_ids = metadata.get("child_bundle_ids") if isinstance(metadata, dict) else None
            if isinstance(child_ids, list) and child_ids:
                first_child = child_ids[0]
                if isinstance(first_child, str):
                    child_update_id = self._extract_update_id(first_child)
                    if child_update_id is not None:
                        return child_update_id
        if not bundle_id.startswith("telegram_"):
            return None
        suffix = bundle_id.removeprefix("telegram_")
        return int(suffix) if suffix.isdigit() else None

    def _max_update_id(self, bundle_id: str) -> int | None:
        source_bundle = self._load_source_bundle(bundle_id)
        if source_bundle is not None:
            metadata = source_bundle.metadata
            child_ids = metadata.get("child_bundle_ids") if isinstance(metadata, dict) else None
            if isinstance(child_ids, list) and child_ids:
                update_ids = [
                    update_id
                    for child_id in child_ids
                    if isinstance(child_id, str)
                    for update_id in [self._extract_update_id(child_id)]
                    if update_id is not None
                ]
                if update_ids:
                    return max(update_ids)
        return self._extract_update_id(bundle_id)

    def _commit_processed_bundle(self, bundle_id: str) -> None:
        update_id = self._max_update_id(bundle_id)
        if update_id is not None:
            self._state_store.set_last_update_id(update_id)

    def _group_bundle_ids(self, bundle_ids: list[str]) -> list[str]:
        grouped_ids: list[str] = []
        index = 0
        while index < len(bundle_ids):
            current_id = bundle_ids[index]
            current_bundle = self._load_source_bundle(current_id)
            if current_bundle is None:
                grouped_ids.append(current_id)
                index += 1
                continue

            next_id = bundle_ids[index + 1] if index + 1 < len(bundle_ids) else None
            if next_id is None:
                grouped_ids.append(current_id)
                index += 1
                continue

            next_bundle = self._load_source_bundle(next_id)
            if next_bundle is None or not self._should_group(current_bundle, next_bundle):
                grouped_ids.append(current_id)
                index += 1
                continue

            grouped_ids.append(self._materialize_group([current_bundle, next_bundle]))
            index += 2
        return grouped_ids

    def _load_source_bundle(self, bundle_id: str) -> SourceBundle | None:
        raw_payload = self._raw_store.load_raw_payload(bundle_id)
        if raw_payload is None:
            return None
        return SourceBundle.model_validate(raw_payload)

    def _should_group(self, left: SourceBundle, right: SourceBundle) -> bool:
        left_metadata = left.metadata if isinstance(left.metadata, dict) else {}
        right_metadata = right.metadata if isinstance(right.metadata, dict) else {}
        left_chat = left_metadata.get("chat")
        right_chat = right_metadata.get("chat")
        if not isinstance(left_chat, dict) or not isinstance(right_chat, dict):
            return False
        if left_chat.get("id") != right_chat.get("id"):
            return False
        left_sender = self._sender_id(left_metadata)
        right_sender = self._sender_id(right_metadata)
        if left_sender is None or right_sender is None or left_sender != right_sender:
            return False
        left_date = left_metadata.get("date")
        right_date = right_metadata.get("date")
        if not isinstance(left_date, int) or not isinstance(right_date, int):
            return False
        if not 0 <= right_date - left_date <= 90:
            return False
        left_has_link = bool(left.canonical_url or left.links)
        right_has_link = bool(right.canonical_url or right.links)
        right_has_text = bool("\n".join(right.text_blocks).strip())
        return left_has_link and (not right_has_link) and right_has_text

    def _sender_id(self, metadata: dict[str, object]) -> int | None:
        raw_message = metadata.get("raw_message")
        if not isinstance(raw_message, dict):
            return None
        sender = raw_message.get("from")
        if not isinstance(sender, dict):
            return None
        sender_id = sender.get("id")
        return sender_id if isinstance(sender_id, int) else None

    def _materialize_group(self, bundles: list[SourceBundle]) -> str:
        first_bundle = bundles[0]
        last_bundle = bundles[-1]
        grouped_id = f"{first_bundle.bundle_id}_group_{last_bundle.bundle_id.removeprefix('telegram_')}"
        grouped_bundle = SourceBundle(
            bundle_id=grouped_id,
            source=first_bundle.source,
            source_type="message_group",
            canonical_url=first_bundle.canonical_url,
            title=first_bundle.title or last_bundle.title,
            text_blocks=[block for bundle in bundles for block in bundle.text_blocks],
            links=first_bundle.links,
            asset_paths=[path for bundle in bundles for path in bundle.asset_paths],
            image_urls=[url for bundle in bundles for url in bundle.image_urls],
            metadata={
                "child_bundle_ids": [bundle.bundle_id for bundle in bundles],
                "chat": first_bundle.metadata.get("chat") if isinstance(first_bundle.metadata, dict) else {},
                "date": first_bundle.metadata.get("date") if isinstance(first_bundle.metadata, dict) else None,
                "message_id": first_bundle.metadata.get("message_id") if isinstance(first_bundle.metadata, dict) else None,
                "reply_target_message_id": last_bundle.metadata.get("message_id") if isinstance(last_bundle.metadata, dict) else None,
            },
        )
        self._raw_store.save_raw_payload(grouped_id, grouped_bundle.model_dump(mode="json"))
        return grouped_id

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
        graph.add_node("backfill_full_answers", self._backfill_full_answers_node)
        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "index")
        graph.add_edge("index", "generate")
        graph.add_edge("generate", "store_answers")
        graph.add_edge("store_answers", "reply")
        graph.add_edge("reply", "backfill_full_answers")
        graph.add_edge("backfill_full_answers", END)
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
        success, reason, answer_path, markdown_path = workflow.run(state["bundle_id"], quick=True)
        if not success:
            raise RuntimeError(reason)
        return {
            "answer_path": answer_path or "",
            "markdown_path": markdown_path or "",
        }

    def _backfill_full_answers_node(self, state: BundleProcessState) -> dict[str, object]:
        log_dir = Path(self._settings.sqlite_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "full-answer-backfill.log"
        log_file = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "xhs_interview_answer_copilot.main",
            "backfill-full-answers",
            state["bundle_id"],
        ]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src{os.pathsep}{existing_pythonpath}"
        subprocess.Popen(
            command,
            stdout=log_file,
            stderr=log_file,
            env=env,
            start_new_session=True,
        )
        log_file.close()
        return {"backfill_started": True}

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
