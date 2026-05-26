from __future__ import annotations

import math
import re
from collections import Counter

from knowledge_agent.config import Settings
from knowledge_agent.storage.vector_store import IndexedQuestion, QuestionVectorStore
from knowledge_agent.workflows.openai_clients import build_embeddings_model

VALID_RETRIEVAL_MODES = {"vector", "bm25", "hybrid"}


class QuestionRetriever:
    def __init__(
        self,
        settings: Settings,
        vector_store: QuestionVectorStore,
        retrieval_mode: str | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._retrieval_mode = self._normalize_mode(retrieval_mode or settings.retrieval_mode)

    def search(
        self,
        query: str,
        top_k: int = 3,
        exclude_note_id: str | None = None,
    ) -> tuple[bool, str, list[IndexedQuestion]]:
        try:
            if self._retrieval_mode == "bm25":
                results = self._search_bm25(query, top_k=top_k, exclude_note_id=exclude_note_id)
            elif self._retrieval_mode == "hybrid":
                results = self._search_hybrid(query, top_k=top_k, exclude_note_id=exclude_note_id)
            else:
                results = self._search_vector(query, top_k=top_k, exclude_note_id=exclude_note_id)
        except Exception as exc:
            return False, f"Retrieval failed: {exc}", []
        return True, f"Retrieval completed with mode={self._retrieval_mode}.", results

    def _normalize_mode(self, mode: str) -> str:
        normalized_mode = mode.strip().lower()
        return normalized_mode if normalized_mode in VALID_RETRIEVAL_MODES else "vector"

    def _search_vector(
        self,
        query: str,
        *,
        top_k: int,
        exclude_note_id: str | None,
    ) -> list[IndexedQuestion]:
        if (
            not self._settings.embedding_model.startswith("local-hash")
            and self._settings.openai_api_key is None
            and self._settings.openai_base_url is None
        ):
            raise RuntimeError("Configure OPENAI_API_KEY or OPENAI_BASE_URL first.")
        embeddings = build_embeddings_model(self._settings)
        query_embedding = embeddings.embed_query(query)
        return self._vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            embedding_model=self._settings.embedding_model,
            min_score=self._settings.retrieval_min_score,
            exclude_note_id=exclude_note_id,
        )

    def _search_bm25(
        self,
        query: str,
        *,
        top_k: int,
        exclude_note_id: str | None,
    ) -> list[IndexedQuestion]:
        candidates = self._vector_store.list_indexed_questions(exclude_note_id=exclude_note_id)
        return self._score_bm25(query=query, candidates=candidates)[:top_k]

    def _search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        exclude_note_id: str | None,
    ) -> list[IndexedQuestion]:
        vector_results = self._search_vector(query, top_k=max(top_k * 4, top_k), exclude_note_id=exclude_note_id)
        bm25_results = self._search_bm25(query, top_k=max(top_k * 4, top_k), exclude_note_id=exclude_note_id)
        combined: dict[str, IndexedQuestion] = {item.record_id: item for item in vector_results}
        combined.update({item.record_id: item for item in bm25_results})
        vector_ranks = {item.record_id: rank for rank, item in enumerate(vector_results, start=1)}
        bm25_ranks = {item.record_id: rank for rank, item in enumerate(bm25_results, start=1)}
        merged = []
        for item in combined.values():
            score = self._rrf_score(vector_ranks.get(item.record_id), bm25_ranks.get(item.record_id))
            if score <= 0:
                continue
            merged.append(self._with_score(item, score))
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def _score_bm25(self, *, query: str, candidates: list[IndexedQuestion]) -> list[IndexedQuestion]:
        query_terms = self._tokenize(query)
        if not query_terms or not candidates:
            return []
        documents = [self._tokenize(self._build_document_text(item)) for item in candidates]
        document_count = len(documents)
        average_length = sum(len(document) for document in documents) / max(document_count, 1)
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))

        k1 = 1.5
        b = 0.75
        scored: list[IndexedQuestion] = []
        for item, document in zip(candidates, documents):
            term_frequency = Counter(document)
            document_length = len(document) or 1
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency <= 0:
                    continue
                idf = math.log(1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                denominator = frequency + k1 * (1 - b + b * document_length / max(average_length, 1))
                score += idf * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append(self._with_score(item, score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return self._normalize_and_filter_scores(scored)

    def _build_document_text(self, item: IndexedQuestion) -> str:
        return "\n".join(
            [
                item.title,
                item.summary,
                item.question,
                item.category,
                " ".join(item.keywords),
            ]
        )

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        latin_terms = re.findall(r"[a-z0-9_+.#-]+", normalized)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
        chinese_bigrams = ["".join(pair) for pair in zip(chinese_chars, chinese_chars[1:])]
        return latin_terms + chinese_chars + chinese_bigrams

    def _normalize_and_filter_scores(self, items: list[IndexedQuestion]) -> list[IndexedQuestion]:
        if not items:
            return []
        max_score = max(item.score for item in items)
        if max_score <= 0:
            return []
        normalized_items = [self._with_score(item, item.score / max_score) for item in items]
        return [item for item in normalized_items if item.score >= self._settings.retrieval_min_score]

    def _rrf_score(self, vector_rank: int | None, bm25_rank: int | None) -> float:
        k = 60
        score = 0.0
        if vector_rank is not None:
            score += 1.0 / (k + vector_rank)
        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank)
        return score

    def _with_score(self, item: IndexedQuestion, score: float) -> IndexedQuestion:
        return IndexedQuestion(
            record_id=item.record_id,
            note_id=item.note_id,
            note_url=item.note_url,
            title=item.title,
            summary=item.summary,
            question=item.question,
            category=item.category,
            keywords=item.keywords,
            score=score,
        )
