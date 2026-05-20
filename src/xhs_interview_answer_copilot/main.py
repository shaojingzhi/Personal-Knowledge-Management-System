import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from xhs_interview_answer_copilot.config import settings


DEFAULT_TELEGRAM_WORKER_SESSION = "xhs-copilot-telegram"


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _telegram_daemon_log_path() -> Path:
    log_dir = Path(settings.sqlite_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "telegram-daemon.log"


def _telegram_daemon_log_path_readonly() -> Path:
    return Path(settings.sqlite_path).parent / "telegram-daemon.log"


def _tmux_attach_command(session_name: str) -> str:
    return f"tmux attach -t {shlex.quote(session_name)}"


def _start_telegram_worker_in_tmux(
    *,
    session_name: str,
    interval_seconds: int,
    failure_backoff_seconds: int,
) -> tuple[bool, str, Path]:
    log_path = _telegram_daemon_log_path()
    if not _tmux_available():
        return False, "tmux is not installed or not found in PATH.", log_path
    if _tmux_session_exists(session_name):
        return True, "Telegram worker is already running.", log_path

    command_parts = [
        f"PYTHONPATH={shlex.quote('src' + os.pathsep + os.getenv('PYTHONPATH'))}" if os.getenv("PYTHONPATH") else "PYTHONPATH=src",
        shlex.quote(sys.executable),
        "-u",
        "-m",
        "xhs_interview_answer_copilot.main",
        "process-telegram-daemon",
        "--interval-seconds",
        str(interval_seconds),
        "--failure-backoff-seconds",
        str(failure_backoff_seconds),
        ">>",
        shlex.quote(str(log_path)),
        "2>&1",
    ]
    result = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-c",
            str(Path.cwd()),
            " ".join(command_parts),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "Failed to start tmux session.", log_path
    return True, "Telegram worker started.", log_path


def _stop_telegram_worker_in_tmux(session_name: str) -> tuple[bool, str]:
    if not _tmux_available():
        return False, "tmux is not installed or not found in PATH."
    if not _tmux_session_exists(session_name):
        return True, "Telegram worker is not running."
    result = subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "Failed to stop tmux session."
    return True, "Telegram worker stopped."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xhs-copilot")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show current bootstrap configuration")
    subparsers.add_parser(
        "login", help="Open a persistent Xiaohongshu browser profile for login"
    )
    subparsers.add_parser(
        "healthcheck", help="Check whether the stored Xiaohongshu session is valid"
    )
    subparsers.add_parser(
        "discover", help="Scan the configured favorites page and store new note ids"
    )
    extract_parser = subparsers.add_parser(
        "extract-note", help="Extract raw note content for a discovered note id"
    )
    extract_parser.add_argument("note_id", help="The discovered Xiaohongshu note id")
    normalize_parser = subparsers.add_parser(
        "normalize-note", help="Run LangGraph normalization for one extracted note"
    )
    normalize_parser.add_argument("note_id", help="The raw bundle id to normalize")
    index_parser = subparsers.add_parser(
        "index-note", help="Index normalized questions for one note into local vector storage"
    )
    index_parser.add_argument("note_id", help="The discovered Xiaohongshu note id")
    search_parser = subparsers.add_parser(
        "search-similar", help="Search similar indexed interview questions"
    )
    search_parser.add_argument("query", help="Free-text query for similarity retrieval")
    eval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate vector, bm25, and hybrid retrieval modes on a fixed dataset",
    )
    eval_parser.add_argument("dataset_path", help="Path to the retrieval evaluation JSON dataset")
    eval_parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Optional subset of retrieval modes to evaluate: vector bm25 hybrid",
    )
    answer_parser = subparsers.add_parser(
        "generate-answers", help="Generate RAG-based answers for one normalized note"
    )
    answer_parser.add_argument("note_id", help="The discovered Xiaohongshu note id")
    answer_parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate a local quick answer set for fast Telegram replies",
    )
    store_answers_parser = subparsers.add_parser(
        "store-answer-records",
        help="Store generated question-answer records into vector storage",
    )
    store_answers_parser.add_argument("note_id", help="The processed bundle id")
    backfill_parser = subparsers.add_parser(
        "backfill-full-answers",
        help="Regenerate full answers, refresh records, and try to send answer.md to Telegram",
    )
    backfill_parser.add_argument("note_id", help="The processed bundle id")
    reply_parser = subparsers.add_parser(
        "reply-telegram",
        help="Send generated answer text back to Telegram",
    )
    reply_parser.add_argument("note_id", help="The processed bundle id")
    subparsers.add_parser(
        "process-telegram-once",
        help="Ingest Telegram once and run the full LangGraph text pipeline automatically",
    )
    daemon_parser = subparsers.add_parser(
        "process-telegram-daemon",
        help="Continuously poll Telegram and run the full pipeline in a loop",
    )
    daemon_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=15,
        help="Sleep interval after a successful polling loop",
    )
    daemon_parser.add_argument(
        "--failure-backoff-seconds",
        type=int,
        default=45,
        help="Sleep interval after a failed polling loop",
    )
    daemon_parser.add_argument(
        "--max-loops",
        type=int,
        default=None,
        help="Optional loop cap for testing the daemon command",
    )
    worker_start_parser = subparsers.add_parser(
        "telegram-worker-start",
        help="Start the Telegram daemon in a detached tmux session",
    )
    worker_start_parser.add_argument(
        "--session-name",
        default=DEFAULT_TELEGRAM_WORKER_SESSION,
        help="tmux session name for the Telegram worker",
    )
    worker_start_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=15,
        help="Sleep interval after a successful polling loop",
    )
    worker_start_parser.add_argument(
        "--failure-backoff-seconds",
        type=int,
        default=45,
        help="Sleep interval after a failed polling loop",
    )
    worker_status_parser = subparsers.add_parser(
        "telegram-worker-status",
        help="Show whether the tmux-backed Telegram worker is running",
    )
    worker_status_parser.add_argument(
        "--session-name",
        default=DEFAULT_TELEGRAM_WORKER_SESSION,
        help="tmux session name for the Telegram worker",
    )
    worker_stop_parser = subparsers.add_parser(
        "telegram-worker-stop",
        help="Stop the tmux-backed Telegram worker",
    )
    worker_stop_parser.add_argument(
        "--session-name",
        default=DEFAULT_TELEGRAM_WORKER_SESSION,
        help="tmux session name for the Telegram worker",
    )
    feishu_parser = subparsers.add_parser(
        "ingest-feishu-event",
        help="Ingest one Feishu event payload file into a SourceBundle",
    )
    feishu_parser.add_argument("json_path", help="Path to a Feishu event JSON file")
    subparsers.add_parser(
        "ingest-telegram-once",
        help="Fetch Telegram updates once and persist raw message bundles",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "status"):
        from xhs_interview_answer_copilot.storage.telegram_state_store import TelegramStateStore

        retrieval_mode = TelegramStateStore(sqlite_path=settings.sqlite_path).get_retrieval_mode(
            settings.retrieval_mode
        )
        print(
            "XHS Interview Answer Copilot initialized. "
            f"Favorites folder: {settings.xhs_favorites_folder_name}; "
            f"profile dir: {settings.xhs_profile_dir}; "
            f"retrieval mode: {retrieval_mode}"
        )
        return

    if args.command == "login":
        from xhs_interview_answer_copilot.collectors.browser_session import (
            BrowserSessionManager,
        )

        session_manager = BrowserSessionManager(settings=settings)
        result = session_manager.bootstrap_login()
        print(f"authenticated={result.is_authenticated}")
        print(f"reason={result.reason}")
        print(f"current_url={result.current_url}")
        return

    if args.command == "healthcheck":
        from xhs_interview_answer_copilot.collectors.browser_session import (
            BrowserSessionManager,
        )

        session_manager = BrowserSessionManager(settings=settings)
        result = session_manager.check_authentication()
        print(f"authenticated={result.is_authenticated}")
        print(f"reason={result.reason}")
        print(f"current_url={result.current_url}")
        return

    if args.command == "discover":
        from xhs_interview_answer_copilot.collectors.favorites_collector import (
            FavoritesCollector,
        )
        from xhs_interview_answer_copilot.storage.discovery_store import DiscoveryStore

        collector = FavoritesCollector(
            settings=settings,
            store=DiscoveryStore(sqlite_path=settings.sqlite_path),
        )
        result = collector.discover_new_notes()
        print(f"reason={result.reason}")
        print(f"scanned_links={result.scanned_links}")
        print(f"new_notes={len(result.discovered_notes)}")
        for note in result.discovered_notes:
            print(f"note_id={note.note_id} note_url={note.note_url}")
        return

    if args.command == "extract-note":
        from xhs_interview_answer_copilot.collectors.note_extractor import NoteExtractor
        from xhs_interview_answer_copilot.storage.discovery_store import DiscoveryStore
        from xhs_interview_answer_copilot.storage.raw_artifact_store import (
            RawArtifactStore,
        )

        extractor = NoteExtractor(
            settings=settings,
            discovery_store=DiscoveryStore(sqlite_path=settings.sqlite_path),
            raw_store=RawArtifactStore(output_dir=settings.output_dir),
        )
        result = extractor.extract_note(args.note_id)
        print(f"success={result.success}")
        print(f"reason={result.reason}")
        print(f"raw_path={result.raw_path}")
        if result.note is not None:
            print(f"title={result.note.title}")
            print(f"image_count={len(result.note.image_urls)}")
        return

    if args.command == "normalize-note":
        from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
        from xhs_interview_answer_copilot.workflows.normalize_note_workflow import (
            NormalizeNoteWorkflow,
        )

        workflow = NormalizeNoteWorkflow(
            settings=settings,
            raw_store=RawArtifactStore(output_dir=settings.output_dir),
            normalized_store=NormalizedArtifactStore(output_dir=settings.output_dir),
        )
        success, reason, normalized_path = workflow.run(args.note_id)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"normalized_path={normalized_path}")
        return

    if args.command == "index-note":
        from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.index_note_workflow import (
            IndexNoteWorkflow,
        )

        workflow = IndexNoteWorkflow(
            settings=settings,
            normalized_store=NormalizedArtifactStore(output_dir=settings.output_dir),
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
        )
        success, reason, indexed_count = workflow.run(args.note_id)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"indexed_count={indexed_count}")
        return

    if args.command == "search-similar":
        from xhs_interview_answer_copilot.storage.telegram_state_store import TelegramStateStore
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.retrieve_questions import (
            QuestionRetriever,
        )

        state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)
        retriever = QuestionRetriever(
            settings=settings,
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
            retrieval_mode=state_store.get_retrieval_mode(settings.retrieval_mode),
        )
        success, reason, results = retriever.search(
            args.query,
            top_k=settings.retrieval_top_k,
        )
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"results={len(results)}")
        for result in results:
            print(
                f"score={result.score:.4f} note_id={result.note_id} "
                f"category={result.category} question={result.question}"
            )
        return

    if args.command == "eval-retrieval":
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.eval_retrieval_workflow import (
            EvalRetrievalWorkflow,
        )

        workflow = EvalRetrievalWorkflow(
            settings=settings,
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
        )
        result = workflow.run(args.dataset_path, modes=args.modes)
        print(f"success={result.success}")
        print(f"reason={result.reason}")
        print(f"report_json_path={result.report_json_path}")
        print(f"report_markdown_path={result.report_markdown_path}")
        if not result.success:
            raise SystemExit(1)
        return

    if args.command == "generate-answers":
        from xhs_interview_answer_copilot.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.generate_answers_workflow import (
            GenerateAnswersWorkflow,
        )

        workflow = GenerateAnswersWorkflow(
            settings=settings,
            normalized_store=NormalizedArtifactStore(output_dir=settings.output_dir),
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
            answer_store=AnswerArtifactStore(output_dir=settings.output_dir),
        )
        success, reason, answer_path, markdown_path = workflow.run(args.note_id, quick=args.quick)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"answer_path={answer_path}")
        print(f"markdown_path={markdown_path}")
        return

    if args.command == "store-answer-records":
        from xhs_interview_answer_copilot.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.store_answer_records_workflow import (
            StoreAnswerRecordsWorkflow,
        )

        workflow = StoreAnswerRecordsWorkflow(
            settings=settings,
            answer_store=AnswerArtifactStore(output_dir=settings.output_dir),
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
        )
        success, reason, stored_count = workflow.run(args.note_id)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"stored_count={stored_count}")
        return

    if args.command == "backfill-full-answers":
        from xhs_interview_answer_copilot.dispatch.telegram_dispatcher import (
            TelegramDispatcher,
        )
        from xhs_interview_answer_copilot.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
        from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.generate_answers_workflow import (
            GenerateAnswersWorkflow,
        )
        from xhs_interview_answer_copilot.workflows.store_answer_records_workflow import (
            StoreAnswerRecordsWorkflow,
        )

        answer_store = AnswerArtifactStore(output_dir=settings.output_dir)
        vector_store = QuestionVectorStore(sqlite_path=settings.sqlite_path)
        answer_workflow = GenerateAnswersWorkflow(
            settings=settings,
            normalized_store=NormalizedArtifactStore(output_dir=settings.output_dir),
            vector_store=vector_store,
            answer_store=answer_store,
        )
        success, reason, answer_path, markdown_path = answer_workflow.run(args.note_id)
        print(f"answer_success={success}")
        print(f"answer_reason={reason}")
        print(f"answer_path={answer_path}")
        print(f"markdown_path={markdown_path}")
        if not success:
            raise SystemExit(1)
        store_workflow = StoreAnswerRecordsWorkflow(
            settings=settings,
            answer_store=answer_store,
            vector_store=vector_store,
        )
        store_success, store_reason, stored_count = store_workflow.run(args.note_id)
        print(f"store_success={store_success}")
        print(f"store_reason={store_reason}")
        print(f"stored_count={stored_count}")
        dispatcher = TelegramDispatcher(
            settings=settings,
            answer_store=answer_store,
            raw_store=RawArtifactStore(output_dir=settings.output_dir),
        )
        document_caption = "完整答案已生成，Markdown 文件见附件。"
        if not store_success:
            document_caption = "完整答案已生成，Markdown 文件见附件。注意：本地答案记录入库刷新失败。"
        document_result = dispatcher.send_answer_markdown_document(
            args.note_id,
            document_caption,
        )
        print(f"document_success={document_result.success}")
        print(f"document_reason={document_result.reason}")
        print(f"document_message_id={document_result.message_id}")
        if not document_result.success:
            status_result = dispatcher.send_status_message(
                args.note_id,
                f"完整答案已生成，但 Markdown 文件发送失败：{document_result.reason}",
            )
            print(f"document_failure_notice_success={status_result.success}")
            print(f"document_failure_notice_reason={status_result.reason}")
        return

    if args.command == "reply-telegram":
        from xhs_interview_answer_copilot.dispatch.telegram_dispatcher import (
            TelegramDispatcher,
        )
        from xhs_interview_answer_copilot.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore

        dispatcher = TelegramDispatcher(
            settings=settings,
            answer_store=AnswerArtifactStore(output_dir=settings.output_dir),
            raw_store=RawArtifactStore(output_dir=settings.output_dir),
        )
        result = dispatcher.reply_answers(args.note_id)
        print(f"success={result.success}")
        print(f"reason={result.reason}")
        print(f"message_id={result.message_id}")
        print(f"sent_count={result.sent_count}")
        return

    if args.command == "process-telegram-once":
        from xhs_interview_answer_copilot.workflows.process_telegram_once_workflow import (
            ProcessTelegramOnceWorkflow,
        )

        workflow = ProcessTelegramOnceWorkflow(settings=settings)
        success, reason, processed = workflow.run_once()
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"processed_bundles={processed}")
        return

    if args.command == "process-telegram-daemon":
        from xhs_interview_answer_copilot.workflows.process_telegram_daemon_workflow import (
            ProcessTelegramDaemonWorkflow,
        )

        workflow = ProcessTelegramDaemonWorkflow(settings=settings)
        result = workflow.run(
            interval_seconds=args.interval_seconds,
            failure_backoff_seconds=args.failure_backoff_seconds,
            max_loops=args.max_loops,
        )
        print(f"success={result.success}")
        print(f"reason={result.reason}")
        print(f"loops={result.loops}")
        print(f"processed_bundles={result.processed_bundles}")
        return

    if args.command == "telegram-worker-start":
        success, reason, log_path = _start_telegram_worker_in_tmux(
            session_name=args.session_name,
            interval_seconds=args.interval_seconds,
            failure_backoff_seconds=args.failure_backoff_seconds,
        )
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"session_name={args.session_name}")
        print(f"log_path={log_path}")
        print(f"attach_command={_tmux_attach_command(args.session_name)}")
        return

    if args.command == "telegram-worker-status":
        log_path = _telegram_daemon_log_path_readonly()
        running = _tmux_available() and _tmux_session_exists(args.session_name)
        print(f"running={running}")
        print(f"session_name={args.session_name}")
        print(f"log_path={log_path}")
        print(f"attach_command={_tmux_attach_command(args.session_name)}")
        return

    if args.command == "telegram-worker-stop":
        success, reason = _stop_telegram_worker_in_tmux(args.session_name)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"session_name={args.session_name}")
        return

    if args.command == "ingest-feishu-event":
        from xhs_interview_answer_copilot.collectors.feishu_ingestor import FeishuIngestor
        from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore

        ingestor = FeishuIngestor(raw_store=RawArtifactStore(output_dir=settings.output_dir))
        result = ingestor.ingest_event_file(args.json_path)
        print(f"success={result.success}")
        print(f"reason={result.reason}")
        print(f"bundle_id={result.bundle_id}")
        print(f"challenge={result.challenge}")
        return

    if args.command == "ingest-telegram-once":
        from xhs_interview_answer_copilot.collectors.telegram_ingestor import (
            TelegramIngestor,
        )
        from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
        from xhs_interview_answer_copilot.storage.telegram_state_store import (
            TelegramStateStore,
        )

        ingestor = TelegramIngestor(
            settings=settings,
            raw_store=RawArtifactStore(output_dir=settings.output_dir),
            state_store=TelegramStateStore(sqlite_path=settings.sqlite_path),
        )
        result = ingestor.ingest_once()
        print(f"processed_updates={result.processed_updates}")
        print(f"saved_bundles={result.saved_bundles}")
        print(f"bundle_ids={result.bundle_ids}")
        print(f"reason={result.reason}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
