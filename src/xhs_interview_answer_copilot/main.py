import argparse

from xhs_interview_answer_copilot.config import settings


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
    answer_parser = subparsers.add_parser(
        "generate-answers", help="Generate RAG-based answers for one normalized note"
    )
    answer_parser.add_argument("note_id", help="The discovered Xiaohongshu note id")
    subparsers.add_parser(
        "ingest-telegram-once",
        help="Fetch Telegram updates once and persist raw message bundles",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "status"):
        print(
            "XHS Interview Answer Copilot initialized. "
            f"Favorites folder: {settings.xhs_favorites_folder_name}; "
            f"profile dir: {settings.xhs_profile_dir}"
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
        from xhs_interview_answer_copilot.storage.vector_store import QuestionVectorStore
        from xhs_interview_answer_copilot.workflows.retrieve_questions import (
            QuestionRetriever,
        )

        retriever = QuestionRetriever(
            settings=settings,
            vector_store=QuestionVectorStore(sqlite_path=settings.sqlite_path),
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
        success, reason, answer_path, markdown_path = workflow.run(args.note_id)
        print(f"success={success}")
        print(f"reason={reason}")
        print(f"answer_path={answer_path}")
        print(f"markdown_path={markdown_path}")
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
        print(f"reason={result.reason}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
