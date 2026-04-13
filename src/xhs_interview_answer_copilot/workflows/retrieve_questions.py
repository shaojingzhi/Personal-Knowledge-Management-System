from __future__ import annotations

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.vector_store import IndexedQuestion, QuestionVectorStore
from xhs_interview_answer_copilot.workflows.openai_clients import build_embeddings_model


class QuestionRetriever:
    def __init__(self, settings: Settings, vector_store: QuestionVectorStore) -> None:
        self._settings = settings
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 3,
        exclude_note_id: str | None = None,
    ) -> tuple[bool, str, list[IndexedQuestion]]:
        if self._settings.openai_api_key is None and self._settings.openai_base_url is None:
            return False, "Configure OPENAI_API_KEY or OPENAI_BASE_URL first.", []
        try:
            embeddings = build_embeddings_model(self._settings)
            query_embedding = embeddings.embed_query(query)
            results = self._vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                embedding_model=self._settings.embedding_model,
                min_score=self._settings.retrieval_min_score,
                exclude_note_id=exclude_note_id,
            )
        except Exception as exc:
            return False, f"Retrieval failed: {exc}", []
        return True, "Retrieval completed.", results
