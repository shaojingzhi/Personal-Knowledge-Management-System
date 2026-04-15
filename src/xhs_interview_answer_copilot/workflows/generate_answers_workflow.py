from __future__ import annotations

import json
from importlib import import_module
from typing import cast
from typing import NotRequired, TypedDict

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.answer_artifact_store import AnswerArtifactStore
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.vector_store import IndexedQuestion, QuestionVectorStore
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.openai_clients import build_chat_model
from xhs_interview_answer_copilot.workflows.retrieve_questions import QuestionRetriever
from xhs_interview_answer_copilot.workflows.schemas import (
    GeneratedAnswerSet,
    InterviewQuestion,
    NormalizedNote,
)


class AnswerState(TypedDict):
    note_id: str
    normalized_note: NormalizedNote
    retrieved_context: NotRequired[dict[str, list[IndexedQuestion]]]
    answer_set: NotRequired[GeneratedAnswerSet]
    answer_path: NotRequired[str]
    markdown_path: NotRequired[str]


class GenerateAnswersWorkflow:
    _BATCH_SIZE = 4

    def __init__(
        self,
        settings: Settings,
        normalized_store: NormalizedArtifactStore,
        vector_store: QuestionVectorStore,
        answer_store: AnswerArtifactStore,
    ) -> None:
        self._settings = settings
        self._normalized_store = normalized_store
        self._vector_store = vector_store
        self._answer_store = answer_store

    def run(self, note_id: str) -> tuple[bool, str, str | None, str | None]:
        normalized_note = self._normalized_store.load_normalized_note(note_id)
        if normalized_note is None:
            return False, f"Normalized note not found for note_id: {note_id}", None, None
        if not normalized_note.questions:
            return False, "No normalized questions found for answer generation.", None, None
        if self._settings.openai_api_key is None and self._settings.openai_base_url is None:
            return False, "Configure OPENAI_API_KEY or OPENAI_BASE_URL first.", None, None

        try:
            graph_module = import_module("langgraph.graph")
            END = graph_module.END
            START = graph_module.START
            StateGraph = graph_module.StateGraph

            graph = StateGraph(AnswerState)
            graph.add_node("retrieve", self._retrieve_node)
            graph.add_node("generate", self._generate_node)
            graph.add_node("save", self._save_node)
            graph.add_edge(START, "retrieve")
            graph.add_edge("retrieve", "generate")
            graph.add_edge("generate", "save")
            graph.add_edge("save", END)
            app = graph.compile()
            result = app.invoke({"note_id": note_id, "normalized_note": normalized_note})
        except Exception as exc:
            return False, f"Answer generation failed: {exc}", None, None
        return (
            True,
            "Answer generation completed.",
            result.get("answer_path"),
            result.get("markdown_path"),
        )

    def _retrieve_node(self, state: AnswerState) -> dict[str, object]:
        retriever = QuestionRetriever(self._settings, self._vector_store)
        contexts: dict[str, list[IndexedQuestion]] = {}
        for question in state["normalized_note"].questions:
            success, reason, results = retriever.search(
                query=question.question,
                top_k=self._settings.retrieval_top_k,
                exclude_note_id=state["normalized_note"].note_id,
            )
            if not success:
                raise RuntimeError(reason)
            contexts[question.question] = results
        return {"retrieved_context": contexts}

    def _generate_node(self, state: AnswerState) -> dict[str, object]:
        output_parsers_module = import_module("langchain_core.output_parsers")
        prompts_module = import_module("langchain_core.prompts")
        ChatPromptTemplate = prompts_module.ChatPromptTemplate
        PydanticOutputParser = output_parsers_module.PydanticOutputParser

        llm = build_chat_model(
            settings=self._settings,
            model_name=self._settings.answer_model,
            temperature=0,
        )
        parser = PydanticOutputParser(pydantic_object=GeneratedAnswerSet)

        normalized_note = state["normalized_note"]
        retrieved_context = state.get("retrieved_context", {})
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You generate interview answers from normalized questions. Use retrieved historical context only when it is relevant, and do not fabricate source usage. If a question is algorithmic, include a Python reference implementation in the code field. For non-algorithm questions, leave the code field empty.",
                ),
                (
                    "human",
                    "Generate interview answers for this normalized note.\n"
                    "Return JSON only.\n"
                    "{format_instructions}\n"
                    "note_id: {note_id}\n"
                    "note_url: {note_url}\n"
                    "title: {title}\n"
                    "summary: {summary}\n"
                    "questions: {questions}\n"
                    "retrieved_context: {retrieved_context}",
                ),
            ]
        )
        chain = prompt | llm
        all_answers = []
        for batch in self._iter_question_batches(normalized_note.questions):
            response = chain.invoke(
                {
                    "format_instructions": parser.get_format_instructions(),
                    "note_id": normalized_note.note_id,
                    "note_url": normalized_note.note_url,
                    "title": normalized_note.title,
                    "summary": normalized_note.summary,
                    "questions": json.dumps(
                        [question.model_dump(mode="json") for question in batch],
                        ensure_ascii=False,
                    ),
                    "retrieved_context": self._format_retrieved_context(batch, retrieved_context),
                }
            )
            partial_answer_set = parse_pydantic_response(parser, response)
            all_answers.extend(self._validate_batch_answers(batch, partial_answer_set))
        answer_set = GeneratedAnswerSet(
            note_id=normalized_note.note_id,
            note_url=normalized_note.note_url,
            title=normalized_note.title,
            answers=all_answers,
        )
        valid_source_ids = {
            item.record_id
            for results in retrieved_context.values()
            for item in results
        }
        for answer in answer_set.answers:
            answer.source_ids = [
                source_id for source_id in answer.source_ids if source_id in valid_source_ids
            ]
        return {"answer_set": answer_set}

    def _iter_question_batches(
        self, questions: list[InterviewQuestion]
    ) -> list[list[InterviewQuestion]]:
        return [
            questions[index : index + self._BATCH_SIZE]
            for index in range(0, len(questions), self._BATCH_SIZE)
        ]

    def _format_retrieved_context(
        self,
        questions: list[InterviewQuestion],
        retrieved_context: dict[str, list[IndexedQuestion]],
    ) -> str:
        payload = []
        for question in questions:
            payload.append(
                {
                    "question": question.question,
                    "matches": [
                        {
                            "source_id": item.record_id,
                            "score": round(item.score, 4),
                            "similar_question": item.question,
                            "summary": item.summary,
                        }
                        for item in retrieved_context.get(question.question, [])
                    ],
                }
            )
        return json.dumps(payload, ensure_ascii=False)

    def _validate_batch_answers(
        self,
        questions: list[InterviewQuestion],
        answer_set: GeneratedAnswerSet,
    ) -> list:
        if len(answer_set.answers) != len(questions):
            raise RuntimeError("Answer batch size does not match question batch size")
        answers_by_question = {}
        for answer in answer_set.answers:
            if answer.question in answers_by_question:
                raise RuntimeError(f"Duplicate answer returned for question: {answer.question}")
            answers_by_question[answer.question] = answer
        ordered_answers = []
        for question in questions:
            matched_answer = answers_by_question.get(question.question)
            if matched_answer is None:
                raise RuntimeError(f"Missing answer for question: {question.question}")
            ordered_answers.append(matched_answer)
        return ordered_answers

    def _save_node(self, state: AnswerState) -> dict[str, str]:
        answer_set = cast(GeneratedAnswerSet, state.get("answer_set"))
        answer_path = self._answer_store.save_answers(
            note_id=state["note_id"],
            payload=answer_set.model_dump(mode="json"),
        )
        markdown_path = self._answer_store.save_markdown(
            note_id=state["note_id"],
            markdown=self._build_markdown(answer_set),
        )
        return {"answer_path": str(answer_path), "markdown_path": str(markdown_path)}

    def _build_markdown(self, answer_set: GeneratedAnswerSet) -> str:
        lines = [
            f"# {answer_set.title or answer_set.note_id}",
            "",
            f"- Source ID: `{answer_set.note_id}`",
            f"- Source URL: {answer_set.note_url or 'N/A'}",
            f"- Question Count: {len(answer_set.answers)}",
            "",
        ]
        for index, answer in enumerate(answer_set.answers, start=1):
            lines.extend(
                [
                    f"## Q{index}. {answer.question}",
                    "",
                    "**Short Answer**",
                    "",
                    answer.short_answer,
                    "",
                    "**Detailed Answer**",
                    "",
                    answer.long_answer,
                    "",
                ]
            )
            if answer.code.strip():
                lines.extend(
                    [
                        "**Python Reference Code**",
                        "",
                        "```python",
                        answer.code,
                        "```",
                        "",
                    ]
                )
            if answer.source_ids:
                lines.extend(
                    [
                        "**Grounding Source IDs**",
                        "",
                        ", ".join(f"`{source_id}`" for source_id in answer.source_ids),
                        "",
                    ]
                )
        return "\n".join(lines).strip() + "\n"
