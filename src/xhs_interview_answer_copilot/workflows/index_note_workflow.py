from __future__ import annotations

from importlib import import_module

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.vector_store import (
    QuestionVectorRecord,
    QuestionVectorStore,
)
from xhs_interview_answer_copilot.workflows.openai_clients import build_embeddings_model
from xhs_interview_answer_copilot.workflows.schemas import InterviewQuestion, NormalizedNote


class IndexNoteWorkflow:
    def __init__(
        self,
        settings: Settings,
        normalized_store: NormalizedArtifactStore,
        vector_store: QuestionVectorStore,
    ) -> None:
        self._settings = settings
        self._normalized_store = normalized_store
        self._vector_store = vector_store

    def run(self, note_id: str) -> tuple[bool, str, int]:
        normalized_note = self._normalized_store.load_normalized_note(note_id)
        if normalized_note is None:
            return False, f"Normalized note not found for note_id: {note_id}", 0
        if self._settings.openai_api_key is None and self._settings.openai_base_url is None:
            return False, "Configure OPENAI_API_KEY or OPENAI_BASE_URL first.", 0
        if not normalized_note.questions:
            return False, "No normalized questions found to index.", 0

        try:
            embeddings = build_embeddings_model(self._settings)
            texts = [self._build_embedding_text(normalized_note, question) for question in normalized_note.questions]
            vectors = embeddings.embed_documents(texts)
        except Exception as exc:
            return False, f"Indexing failed: {exc}", 0

        records = [
            QuestionVectorRecord(
                record_id=f"{normalized_note.note_id}:{index}",
                note_id=normalized_note.note_id,
                note_url=normalized_note.note_url,
                title=normalized_note.title,
                summary=normalized_note.summary,
                question=question.question,
                category=question.category,
                keywords=question.keywords,
                embedding=vector,
                embedding_model=self._settings.embedding_model,
            )
            for index, (question, vector) in enumerate(zip(normalized_note.questions, vectors))
        ]
        self._vector_store.delete_by_note(normalized_note.note_id)
        saved_count = self._vector_store.upsert_records(records)
        return True, "Indexing completed.", saved_count

    def _build_embedding_text(
        self,
        normalized_note: NormalizedNote,
        question: InterviewQuestion,
    ) -> str:
        return (
            f"title: {normalized_note.title}\n"
            f"summary: {normalized_note.summary}\n"
            f"question: {question.question}\n"
            f"category: {question.category}\n"
            f"keywords: {', '.join(question.keywords)}"
        )
