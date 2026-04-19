from __future__ import annotations

import json
from importlib import import_module
from typing import Any, cast
from typing import NotRequired, TypedDict

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.answer_artifact_store import AnswerArtifactStore
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.vector_store import IndexedQuestion, QuestionVectorStore
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.llm_retry import invoke_with_retry
from xhs_interview_answer_copilot.workflows.openai_clients import build_chat_model
from xhs_interview_answer_copilot.workflows.retrieve_questions import QuestionRetriever
from xhs_interview_answer_copilot.workflows.schemas import (
    GeneratedAnswerItem,
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
            all_answers.extend(
                self._generate_batch_answers(
                    chain=chain,
                    parser=parser,
                    normalized_note=normalized_note,
                    questions=batch,
                    retrieved_context=retrieved_context,
                )
            )
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

    def _generate_batch_answers(
        self,
        chain: Any,
        parser: Any,
        normalized_note: NormalizedNote,
        questions: list[InterviewQuestion],
        retrieved_context: dict[str, list[IndexedQuestion]],
    ) -> list:
        try:
            response = invoke_with_retry(
                chain,
                {
                    "format_instructions": parser.get_format_instructions(),
                    "note_id": normalized_note.note_id,
                    "note_url": normalized_note.note_url,
                    "title": normalized_note.title,
                    "summary": normalized_note.summary,
                    "questions": json.dumps(
                        [question.model_dump(mode="json") for question in questions],
                        ensure_ascii=False,
                    ),
                    "retrieved_context": self._format_retrieved_context(questions, retrieved_context),
                },
            )
            partial_answer_set = parse_pydantic_response(parser, response)
            return self._validate_batch_answers(questions, partial_answer_set)
        except Exception as exc:
            if not self._should_fallback_answer_generation(exc):
                raise
            if len(questions) == 1:
                return [self._build_fallback_answer(questions[0])]
            answers = []
            for question in questions:
                answers.extend(
                    self._generate_batch_answers(
                        chain=chain,
                        parser=parser,
                        normalized_note=normalized_note,
                        questions=[question],
                        retrieved_context=retrieved_context,
                    )
                )
            return answers

    def _should_fallback_answer_generation(self, exc: Exception) -> bool:
        message = str(exc)
        if "Invalid json output" in message:
            return True
        if "empty" in message.lower() and "output" in message.lower():
            return True
        if "Missing answer for question" in message:
            return True
        if "Duplicate answer returned for question" in message:
            return True
        if "Answer batch size does not match question batch size" in message:
            return True
        return False

    def _build_fallback_answer(self, question: InterviewQuestion) -> GeneratedAnswerItem:
        short_answer = self._fallback_short_answer(question)
        long_answer = self._fallback_long_answer(question)
        code = self._fallback_code(question)
        return GeneratedAnswerItem(
            question=question.question,
            short_answer=short_answer,
            long_answer=long_answer,
            code=code,
            source_ids=[],
        )

    def _fallback_short_answer(self, question: InterviewQuestion) -> str:
        if question.category == "behavior":
            return "建议按背景、行动、结果三段式回答，突出你的真实经历、关键决策和最终产出。"
        if question.category == "backend":
            return "建议先讲核心原理，再补充实际排查或落地经验，最后说明在项目中的取舍。"
        if question.category == "system-design":
            return "建议从整体架构、关键模块、设计取舍和评估方式四个层面回答。"
        return "建议先解释问题本质，再结合真实项目经验给出结构化回答。"

    def _fallback_long_answer(self, question: InterviewQuestion) -> str:
        if question.category == "behavior":
            return (
                "这个问题更适合用 STAR 或 Problem-Action-Result 的结构回答。"
                "先说明当时的背景和目标，再讲你实际承担了什么角色、做了哪些关键动作。"
                "随后补充结果和量化收益，比如效率提升、稳定性改善或项目推进效果。"
                "最后可以总结你从这件事里沉淀出的经验，让答案更完整。"
            )
        if question.category == "backend":
            return (
                "回答这类问题时建议先讲底层原理，说明相关机制为什么会这样设计。"
                "然后结合常见排查路径或工程实践，例如日志、监控、线程栈、内存快照、配置调优等。"
                "如果你有真实案例，可以重点讲定位思路、根因分析和最终修复方案。"
                "这样既能体现基础扎实，也能体现你在复杂问题里的实战能力。"
            )
        if question.category == "system-design":
            return (
                "回答时可以先给出整体架构或流程拆解，把核心模块和信息流讲清楚。"
                "接着说明关键设计点，例如数据流、状态管理、容错、扩展性、评估指标或成本控制。"
                "如果问题与 Agent、RAG 或大模型工程有关，可以补充你的编排方式、质量评估和线上监控闭环。"
                "最后强调为什么这样设计，以及在真实业务里做过哪些取舍。"
            )
        return (
            "这类问题建议先明确概念，再结合你的项目经验说明你是怎么理解和落地的。"
            "如果涉及工具、流程或能力设计，可以补充使用场景、优缺点和优化方向。"
            "面试时尽量避免只讲定义，最好加入自己做过的实践和复盘。"
            "这样答案会更具体，也更容易体现你的工程判断力。"
        )

    def _fallback_code(self, question: InterviewQuestion) -> str:
        lowered = question.question.lower()
        if any(keyword in lowered for keyword in ["算法", "代码", "实现", "手撕"]):
            return (
                "def solve(*args, **kwargs):\n"
                "    \"\"\"Replace with the concrete algorithm after clarifying the exact problem.\"\"\"\n"
                "    pass\n"
            )
        return ""

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
