from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexedQuestion:
    record_id: str
    note_id: str
    note_url: str
    title: str
    summary: str
    question: str
    category: str
    keywords: list[str]
    score: float


@dataclass(frozen=True)
class QuestionVectorRecord:
    record_id: str
    note_id: str
    note_url: str
    title: str
    summary: str
    question: str
    category: str
    keywords: list[str]
    embedding: list[float]
    embedding_model: str


@dataclass(frozen=True)
class QARecord:
    record_id: str
    note_id: str
    note_url: str
    title: str
    question: str
    short_answer: str
    long_answer: str
    markdown_path: str
    answer_path: str
    embedding: list[float]
    embedding_model: str


class QuestionVectorStore:
    def __init__(self, sqlite_path: str) -> None:
        self._sqlite_path = Path(sqlite_path)
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._sqlite_path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS question_vectors (
                    record_id TEXT PRIMARY KEY,
                    note_id TEXT NOT NULL,
                    note_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    question TEXT NOT NULL,
                    category TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_records (
                    record_id TEXT PRIMARY KEY,
                    note_id TEXT NOT NULL,
                    note_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    short_answer TEXT NOT NULL,
                    long_answer TEXT NOT NULL,
                    markdown_path TEXT NOT NULL,
                    answer_path TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL
                )
                """
            )

    def upsert_records(self, records: list[QuestionVectorRecord]) -> int:
        if not records:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO question_vectors (
                    record_id, note_id, note_url, title, summary, question,
                    category, keywords_json, embedding_json, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.record_id,
                        record.note_id,
                        record.note_url,
                        record.title,
                        record.summary,
                        record.question,
                        record.category,
                        json.dumps(record.keywords, ensure_ascii=False),
                        json.dumps(record.embedding),
                        record.embedding_model,
                    )
                    for record in records
                ],
            )
            return connection.total_changes

    def delete_by_note(self, note_id: str) -> int:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM question_vectors WHERE note_id = ?",
                (note_id,),
            )
            return connection.total_changes

    def upsert_qa_records(self, records: list[QARecord]) -> int:
        if not records:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO qa_records (
                    record_id, note_id, note_url, title, question,
                    short_answer, long_answer, markdown_path, answer_path,
                    embedding_json, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.record_id,
                        record.note_id,
                        record.note_url,
                        record.title,
                        record.question,
                        record.short_answer,
                        record.long_answer,
                        record.markdown_path,
                        record.answer_path,
                        json.dumps(record.embedding),
                        record.embedding_model,
                    )
                    for record in records
                ],
            )
            return connection.total_changes

    def delete_qa_by_note(self, note_id: str) -> int:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM qa_records WHERE note_id = ?",
                (note_id,),
            )
            return connection.total_changes

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int,
        embedding_model: str,
        min_score: float,
        exclude_note_id: str | None = None,
    ) -> list[IndexedQuestion]:
        with self._connect() as connection:
            if exclude_note_id is None:
                rows = connection.execute(
                    """
                    SELECT record_id, note_id, note_url, title, summary, question,
                           category, keywords_json, embedding_json
                    FROM question_vectors
                    WHERE embedding_model = ?
                    """
                    ,
                    (embedding_model,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT record_id, note_id, note_url, title, summary, question,
                           category, keywords_json, embedding_json
                    FROM question_vectors
                    WHERE embedding_model = ? AND note_id != ?
                    """,
                    (embedding_model, exclude_note_id),
                ).fetchall()

        scored: list[IndexedQuestion] = []
        for row in rows:
            score = self._cosine_similarity(query_embedding, json.loads(row[8]))
            if score < min_score:
                continue
            scored.append(
                IndexedQuestion(
                    record_id=row[0],
                    note_id=row[1],
                    note_url=row[2],
                    title=row[3],
                    summary=row[4],
                    question=row[5],
                    category=row[6],
                    keywords=json.loads(row[7]),
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left or not right:
            return -1.0
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return numerator / (left_norm * right_norm)
