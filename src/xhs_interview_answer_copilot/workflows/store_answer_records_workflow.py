from __future__ import annotations

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.answer_artifact_store import AnswerArtifactStore
from xhs_interview_answer_copilot.storage.vector_store import QARecord, QuestionVectorStore
from xhs_interview_answer_copilot.workflows.openai_clients import build_embeddings_model
from xhs_interview_answer_copilot.workflows.schemas import GeneratedAnswerSet


class StoreAnswerRecordsWorkflow:
    def __init__(
        self,
        settings: Settings,
        answer_store: AnswerArtifactStore,
        vector_store: QuestionVectorStore,
    ) -> None:
        self._settings = settings
        self._answer_store = answer_store
        self._vector_store = vector_store

    def run(self, note_id: str) -> tuple[bool, str, int]:
        try:
            answer_set = self._answer_store.load_answers(note_id)
            if answer_set is None:
                return False, f"Answer set not found for note_id: {note_id}", 0
            if not answer_set.answers:
                return False, "No generated answers found to store.", 0

            embeddings = build_embeddings_model(self._settings)
            texts = [self._build_embedding_text(answer_set, index) for index, _ in enumerate(answer_set.answers)]
            vectors = embeddings.embed_documents(texts)
            if len(vectors) != len(answer_set.answers):
                return False, "Embedding result count does not match answer count.", 0
        except Exception as exc:
            return False, f"Storing answer records failed: {exc}", 0

        answer_path = self._answer_store.get_answer_json_path(note_id)
        markdown_path = self._answer_store.get_answer_markdown_path(note_id)
        records = [
            QARecord(
                record_id=f"{note_id}:qa:{index}",
                note_id=answer_set.note_id,
                note_url=answer_set.note_url,
                title=answer_set.title,
                question=answer.question,
                short_answer=answer.short_answer,
                long_answer=answer.long_answer,
                markdown_path=str(markdown_path),
                answer_path=str(answer_path),
                embedding=vector,
                embedding_model=self._settings.embedding_model,
            )
            for index, (answer, vector) in enumerate(zip(answer_set.answers, vectors))
        ]
        self._vector_store.delete_qa_by_note(answer_set.note_id)
        saved_count = self._vector_store.upsert_qa_records(records)
        return True, "Stored answer records completed.", saved_count

    def _build_embedding_text(self, answer_set: GeneratedAnswerSet, index: int) -> str:
        answer = answer_set.answers[index]
        return (
            f"title: {answer_set.title}\n"
            f"question: {answer.question}\n"
            f"short_answer: {answer.short_answer}\n"
            f"long_answer: {answer.long_answer}"
        )
