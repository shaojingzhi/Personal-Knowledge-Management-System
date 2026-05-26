import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from knowledge_agent.config import settings


DEFAULT_TELEGRAM_WORKER_SESSION = "knowledge-agent-telegram"
LEGACY_TELEGRAM_WORKER_SESSION = "xhs-copilot-telegram"


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _resolve_project_path(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


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


def _effective_worker_session(session_name: str) -> str:
    if not _tmux_available():
        return session_name
    if session_name != DEFAULT_TELEGRAM_WORKER_SESSION:
        return session_name
    if _tmux_session_exists(session_name):
        return session_name
    if _tmux_session_exists(LEGACY_TELEGRAM_WORKER_SESSION):
        return LEGACY_TELEGRAM_WORKER_SESSION
    return session_name


def _start_telegram_worker_in_tmux(
    *,
    session_name: str,
    interval_seconds: int,
    failure_backoff_seconds: int,
) -> tuple[bool, str, Path]:
    log_path = _telegram_daemon_log_path()
    if not _tmux_available():
        return False, "tmux is not installed or not found in PATH.", log_path
    legacy_running = (
        session_name == DEFAULT_TELEGRAM_WORKER_SESSION
        and _tmux_session_exists(LEGACY_TELEGRAM_WORKER_SESSION)
    )
    if _tmux_session_exists(session_name) or legacy_running:
        return True, "Telegram worker is already running.", log_path

    command_parts = [
        f"PYTHONPATH={shlex.quote('src' + os.pathsep + os.getenv('PYTHONPATH'))}" if os.getenv("PYTHONPATH") else "PYTHONPATH=src",
        shlex.quote(sys.executable),
        "-u",
        "-m",
        "knowledge_agent.main",
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
    target_session = session_name
    if (
        session_name == DEFAULT_TELEGRAM_WORKER_SESSION
        and not _tmux_session_exists(target_session)
        and _tmux_session_exists(LEGACY_TELEGRAM_WORKER_SESSION)
    ):
        target_session = LEGACY_TELEGRAM_WORKER_SESSION
    if not _tmux_session_exists(target_session):
        return True, "Telegram worker is not running."
    result = subprocess.run(
        ["tmux", "kill-session", "-t", target_session],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "Failed to stop tmux session."
    return True, "Telegram worker stopped."


def build_parser(prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(sys.argv[0]).name or "knowledge-agent")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show current bootstrap configuration")
    project_parser = subparsers.add_parser(
        "project", help="Manage the active project context for project-aware answers"
    )
    project_subparsers = project_parser.add_subparsers(dest="project_command")
    project_use_parser = project_subparsers.add_parser(
        "use", help="Set the active project directory"
    )
    project_use_parser.add_argument("path", help="Path to the project directory")
    project_subparsers.add_parser(
        "current", help="Show the current active project directory"
    )
    project_subparsers.add_parser(
        "refresh", help="Refresh the cached context for the current active project"
    )
    project_subparsers.add_parser("clear", help="Clear the current active project directory")
    subparsers.add_parser(
        "login", help="Open the configured browser profile for login"
    )
    subparsers.add_parser(
        "healthcheck", help="Check whether the stored browser session is valid"
    )
    subparsers.add_parser(
        "discover", help="Scan the configured favorites page and store new note ids"
    )
    extract_parser = subparsers.add_parser(
        "extract-note", help="Extract raw source content for a discovered note id"
    )
    extract_parser.add_argument("note_id", help="The discovered source note id")
    normalize_parser = subparsers.add_parser(
        "normalize-note", help="Run LangGraph normalization for one extracted note"
    )
    normalize_parser.add_argument("note_id", help="The raw bundle id to normalize")
    index_parser = subparsers.add_parser(
        "index-note", help="Index normalized questions for one note into local vector storage"
    )
    index_parser.add_argument("note_id", help="The discovered source note id")
    search_parser = subparsers.add_parser(
        "search-similar", help="Search similar indexed interview questions"
    )
    search_parser.add_argument("query", help="Free-text query for similarity retrieval")
    react_parser = subparsers.add_parser(
        "react-agent-demo",
        help="Run a minimal LangGraph ReAct loop over knowledge search, project context, source scan, and answer tools",
    )
    react_parser.add_argument("question", help="Question for the ReAct demo agent")
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
    answer_parser.add_argument("note_id", help="The discovered source note id")
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
        from knowledge_agent.storage.telegram_state_store import TelegramStateStore

        state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)
        retrieval_mode = state_store.get_retrieval_mode(settings.retrieval_mode)
        active_project_path = state_store.get_active_project_path() or "N/A"
        print(
            "Knowledge Management Agent initialized. "
            f"Favorites folder: {settings.xhs_favorites_folder_name}; "
            f"profile dir: {settings.xhs_profile_dir}; "
            f"retrieval mode: {retrieval_mode}; "
            f"active project: {active_project_path}"
        )
        return

    if args.command == "project":
        from knowledge_agent.storage.project_context_store import (
            ProjectContextStore,
        )
        from knowledge_agent.storage.telegram_state_store import TelegramStateStore
        from knowledge_agent.workflows.project_context_workflow import (
            ProjectContextWorkflow,
        )

        state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)
        if args.project_command == "use":
            resolved_path = _resolve_project_path(args.path)
            if not resolved_path.exists():
                print("success=False")
                print(f"reason=Project path does not exist: {resolved_path}")
                raise SystemExit(1)
            if not resolved_path.is_dir():
                print("success=False")
                print(f"reason=Project path is not a directory: {resolved_path}")
                raise SystemExit(1)
            state_store.set_active_project_path(str(resolved_path))
            print("success=True")
            print("reason=Active project updated.")
            print(f"active_project_path={resolved_path}")
            return
        if args.project_command == "current":
            active_project_path = state_store.get_active_project_path()
            print(f"configured={active_project_path is not None}")
            print(f"active_project_path={active_project_path or ''}")
            return
        if args.project_command == "refresh":
            active_project_path = state_store.get_active_project_path()
            if active_project_path is None:
                print("success=False")
                print("reason=No active project configured.")
                raise SystemExit(1)
            workflow = ProjectContextWorkflow(
                project_context_store=ProjectContextStore(output_dir=settings.output_dir)
            )
            success, reason, _, context_path = workflow.run(
                active_project_path,
                force_refresh=True,
            )
            print(f"success={success}")
            print(f"reason={reason}")
            print(f"active_project_path={active_project_path}")
            print(f"project_context_path={context_path}")
            if not success:
                raise SystemExit(1)
            return
        if args.project_command == "clear":
            state_store.clear_active_project_path()
            print("success=True")
            print("reason=Active project cleared.")
            return
        parser.error("Unknown or missing project subcommand.")

    if args.command == "login":
        from knowledge_agent.collectors.browser_session import (
            BrowserSessionManager,
        )

        session_manager = BrowserSessionManager(settings=settings)
        result = session_manager.bootstrap_login()
        print(f"authenticated={result.is_authenticated}")
        print(f"reason={result.reason}")
        print(f"current_url={result.current_url}")
        return

    if args.command == "healthcheck":
        from knowledge_agent.collectors.browser_session import (
            BrowserSessionManager,
        )

        session_manager = BrowserSessionManager(settings=settings)
        result = session_manager.check_authentication()
        print(f"authenticated={result.is_authenticated}")
        print(f"reason={result.reason}")
        print(f"current_url={result.current_url}")
        return

    if args.command == "discover":
        from knowledge_agent.collectors.favorites_collector import (
            FavoritesCollector,
        )
        from knowledge_agent.storage.discovery_store import DiscoveryStore

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
        from knowledge_agent.collectors.note_extractor import NoteExtractor
        from knowledge_agent.storage.discovery_store import DiscoveryStore
        from knowledge_agent.storage.raw_artifact_store import (
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
        from knowledge_agent.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from knowledge_agent.storage.raw_artifact_store import RawArtifactStore
        from knowledge_agent.workflows.normalize_note_workflow import (
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
        from knowledge_agent.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.index_note_workflow import (
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
        from knowledge_agent.storage.telegram_state_store import TelegramStateStore
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.retrieve_questions import (
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
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.eval_retrieval_workflow import (
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

    if args.command == "react-agent-demo":
        from knowledge_agent.storage.project_context_store import ProjectContextStore
        from knowledge_agent.storage.project_deep_context_store import (
            ProjectDeepContextStore,
        )
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.project_context_workflow import (
            ProjectContextWorkflow,
        )
        from knowledge_agent.workflows.project_deep_scan_workflow import (
            ProjectDeepScanWorkflow,
        )
        from knowledge_agent.workflows.project_subagent_scan_workflow import (
            ProjectSubagentScanWorkflow,
        )
        from knowledge_agent.workflows.react_agent_demo_workflow import (
            ReactAgentDemoWorkflow,
        )

        project_deep_context_store = ProjectDeepContextStore(output_dir=settings.output_dir)
        workflow = ReactAgentDemoWorkflow(
            settings=settings,
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
            project_context_workflow=ProjectContextWorkflow(
                ProjectContextStore(output_dir=settings.output_dir)
            ),
            project_deep_scan_workflow=ProjectDeepScanWorkflow(project_deep_context_store),
            project_subagent_scan_workflow=ProjectSubagentScanWorkflow(
                settings=settings,
                project_deep_context_store=project_deep_context_store,
                local_scan_workflow=ProjectDeepScanWorkflow(project_deep_context_store),
            ),
        )
        success, reason, final_answer, steps = workflow.run(args.question)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"steps_json={json.dumps(steps, ensure_ascii=False)}")
        print(f"final_answer={final_answer}")
        if not success:
            raise SystemExit(1)
        return

    if args.command == "generate-answers":
        from knowledge_agent.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from knowledge_agent.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.generate_answers_workflow import (
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
        from knowledge_agent.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.store_answer_records_workflow import (
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
        from knowledge_agent.dispatch.telegram_dispatcher import (
            TelegramDispatcher,
        )
        from knowledge_agent.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from knowledge_agent.storage.raw_artifact_store import RawArtifactStore
        from knowledge_agent.storage.normalized_artifact_store import (
            NormalizedArtifactStore,
        )
        from knowledge_agent.storage.vector_store import QuestionVectorStore
        from knowledge_agent.workflows.generate_answers_workflow import (
            GenerateAnswersWorkflow,
        )
        from knowledge_agent.workflows.store_answer_records_workflow import (
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
        from knowledge_agent.dispatch.telegram_dispatcher import (
            TelegramDispatcher,
        )
        from knowledge_agent.storage.answer_artifact_store import (
            AnswerArtifactStore,
        )
        from knowledge_agent.storage.raw_artifact_store import RawArtifactStore

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
        from knowledge_agent.workflows.process_telegram_once_workflow import (
            ProcessTelegramOnceWorkflow,
        )

        workflow = ProcessTelegramOnceWorkflow(settings=settings)
        success, reason, processed = workflow.run_once()
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"processed_bundles={processed}")
        return

    if args.command == "process-telegram-daemon":
        from knowledge_agent.workflows.process_telegram_daemon_workflow import (
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
        effective_session = _effective_worker_session(args.session_name)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"session_name={effective_session}")
        print(f"log_path={log_path}")
        print(f"attach_command={_tmux_attach_command(effective_session)}")
        return

    if args.command == "telegram-worker-status":
        log_path = _telegram_daemon_log_path_readonly()
        effective_session = _effective_worker_session(args.session_name)
        legacy_running = (
            _tmux_available()
            and
            args.session_name == DEFAULT_TELEGRAM_WORKER_SESSION
            and _tmux_session_exists(LEGACY_TELEGRAM_WORKER_SESSION)
        )
        running = _tmux_available() and (
            _tmux_session_exists(args.session_name)
            or legacy_running
        )
        print(f"running={running}")
        print(f"session_name={effective_session}")
        print(f"legacy_session_name={LEGACY_TELEGRAM_WORKER_SESSION}")
        print(f"log_path={log_path}")
        print(f"attach_command={_tmux_attach_command(effective_session)}")
        return

    if args.command == "telegram-worker-stop":
        success, reason = _stop_telegram_worker_in_tmux(args.session_name)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"session_name={args.session_name}")
        return

    if args.command == "ingest-feishu-event":
        from knowledge_agent.collectors.feishu_ingestor import FeishuIngestor
        from knowledge_agent.storage.raw_artifact_store import RawArtifactStore

        ingestor = FeishuIngestor(raw_store=RawArtifactStore(output_dir=settings.output_dir))
        result = ingestor.ingest_event_file(args.json_path)
        print(f"success={result.success}")
        print(f"reason={result.reason}")
        print(f"bundle_id={result.bundle_id}")
        print(f"challenge={result.challenge}")
        return

    if args.command == "ingest-telegram-once":
        from knowledge_agent.collectors.telegram_ingestor import (
            TelegramIngestor,
        )
        from knowledge_agent.storage.raw_artifact_store import RawArtifactStore
        from knowledge_agent.storage.telegram_state_store import (
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
