from __future__ import annotations

from datetime import datetime, UTC
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from typing import NotRequired, TypedDict

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.answer_artifact_store import AnswerArtifactStore
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.project_answer_memory_store import (
    ProjectAnswerMemoryStore,
)
from xhs_interview_answer_copilot.storage.project_deep_context_store import (
    ProjectDeepContextStore,
)
from xhs_interview_answer_copilot.storage.project_context_store import ProjectContextStore
from xhs_interview_answer_copilot.storage.telegram_state_store import TelegramStateStore
from xhs_interview_answer_copilot.storage.vector_store import IndexedQuestion, QuestionVectorStore
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.llm_retry import invoke_with_retry
from xhs_interview_answer_copilot.workflows.openai_clients import build_chat_model
from xhs_interview_answer_copilot.workflows.project_deep_scan_workflow import (
    ProjectDeepScanWorkflow,
)
from xhs_interview_answer_copilot.workflows.project_context_workflow import (
    ProjectContextWorkflow,
)
from xhs_interview_answer_copilot.workflows.project_subagent_scan_workflow import (
    ProjectSubagentScanWorkflow,
)
from xhs_interview_answer_copilot.workflows.retrieve_questions import QuestionRetriever
from xhs_interview_answer_copilot.workflows.schemas import (
    GeneratedAnswerItem,
    GeneratedAnswerSet,
    InterviewQuestion,
    NormalizedNote,
    ProjectAnswerMemoryRecord,
    ProjectContext,
    ProjectDeepContext,
)


class AnswerState(TypedDict):
    note_id: str
    normalized_note: NormalizedNote
    project_question_flags: NotRequired[dict[str, bool]]
    project_question_topics: NotRequired[dict[str, str]]
    needs_project_context: NotRequired[bool]
    needs_deep_project_scan: NotRequired[bool]
    needs_general_retrieval: NotRequired[bool]
    active_project_path: NotRequired[str]
    project_context: NotRequired[ProjectContext]
    project_context_warning: NotRequired[str]
    deep_project_contexts: NotRequired[dict[str, ProjectDeepContext]]
    project_answer_memory: NotRequired[dict[str, list[ProjectAnswerMemoryRecord]]]
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
        self._state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)
        self._project_answer_memory_store = ProjectAnswerMemoryStore(output_dir=settings.output_dir)
        self._project_deep_scan_workflow = ProjectDeepScanWorkflow(
            project_deep_context_store=ProjectDeepContextStore(output_dir=settings.output_dir)
        )
        self._project_subagent_scan_workflow = ProjectSubagentScanWorkflow(
            settings=settings,
            project_deep_context_store=ProjectDeepContextStore(output_dir=settings.output_dir),
            local_scan_workflow=self._project_deep_scan_workflow,
        )
        self._project_context_workflow = ProjectContextWorkflow(
            project_context_store=ProjectContextStore(output_dir=settings.output_dir)
        )

    def run(self, note_id: str, *, quick: bool = False) -> tuple[bool, str, str | None, str | None]:
        normalized_note = self._normalized_store.load_normalized_note(note_id)
        if normalized_note is None:
            return False, f"Normalized note not found for note_id: {note_id}", None, None
        if not normalized_note.questions:
            return False, "No normalized questions found for answer generation.", None, None
        if quick:
            answer_set = self._build_quick_answer_set(normalized_note)
            answer_path = self._answer_store.save_answers(
                note_id=note_id,
                payload=answer_set.model_dump(mode="json"),
            )
            markdown_path = self._answer_store.save_markdown(
                note_id=note_id,
                markdown=self._build_markdown(answer_set),
            )
            return (
                True,
                "Quick answer generation completed.",
                str(answer_path),
                str(markdown_path),
            )
        if self._settings.openai_api_key is None and self._settings.openai_base_url is None:
            return False, "Configure OPENAI_API_KEY or OPENAI_BASE_URL first.", None, None

        try:
            graph_module = import_module("langgraph.graph")
            END = graph_module.END
            START = graph_module.START
            StateGraph = graph_module.StateGraph

            graph = StateGraph(AnswerState)
            graph.add_node("plan_context", self._plan_context_node)
            graph.add_node("load_project_context", self._load_project_context_node)
            graph.add_node("maybe_delegate_project_scan", self._maybe_delegate_project_scan_node)
            graph.add_node("load_project_answer_memory", self._load_project_answer_memory_node)
            graph.add_node("retrieve", self._retrieve_node)
            graph.add_node("generate", self._generate_node)
            graph.add_node("save", self._save_node)
            graph.add_edge(START, "plan_context")
            graph.add_edge("plan_context", "load_project_context")
            graph.add_edge("load_project_context", "maybe_delegate_project_scan")
            graph.add_edge("maybe_delegate_project_scan", "load_project_answer_memory")
            graph.add_edge("load_project_answer_memory", "retrieve")
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

    def _plan_context_node(self, state: AnswerState) -> dict[str, object]:
        project_question_flags = {
            question.question: self._is_project_related_question(question.question)
            for question in state["normalized_note"].questions
        }
        project_question_topics = {
            question.question: self._detect_project_topic(question.question)
            for question in state["normalized_note"].questions
            if project_question_flags[question.question]
        }
        return {
            "project_question_flags": project_question_flags,
            "project_question_topics": project_question_topics,
            "needs_project_context": any(project_question_flags.values()),
            "needs_deep_project_scan": any(
                self._needs_deep_project_scan(question)
                for question in state["normalized_note"].questions
                if project_question_flags[question.question]
            ),
            "needs_general_retrieval": True,
        }

    def _load_project_context_node(self, state: AnswerState) -> dict[str, object]:
        if not state.get("needs_project_context"):
            return {}
        active_project_path = self._state_store.get_active_project_path()
        if active_project_path is None:
            return {
                "active_project_path": "",
                "project_context_warning": (
                    "No active project is configured. Do not invent repository-specific implementation details."
                )
            }
        success, reason, project_context, _ = self._project_context_workflow.run(active_project_path)
        if not success or project_context is None:
            return {
                "active_project_path": active_project_path,
                "project_context_warning": (
                    f"Project context is unavailable: {reason}. Do not invent repository-specific implementation details."
                )
            }
        return {"project_context": project_context, "active_project_path": active_project_path}

    def _maybe_delegate_project_scan_node(self, state: AnswerState) -> dict[str, object]:
        if not state.get("needs_deep_project_scan"):
            return {"deep_project_contexts": {}}
        active_project_path = state.get("active_project_path")
        if not active_project_path:
            return {"deep_project_contexts": {}}
        topics = sorted(set(state.get("project_question_topics", {}).values()))
        deep_contexts: dict[str, ProjectDeepContext] = {}
        warning_messages: list[str] = []
        for topic in topics:
            topic_questions = [
                question_text
                for question_text, question_topic in state.get("project_question_topics", {}).items()
                if question_topic == topic
            ]
            joined_question = "\n".join(topic_questions)
            success, reason, deep_context = self._scan_project_topic(
                active_project_path=active_project_path,
                topic=topic,
                question=joined_question,
                project_context=state.get("project_context"),
            )
            if success and deep_context is not None:
                deep_contexts[topic] = deep_context
                continue
            warning_messages.append(f"{topic}: {reason}")
        warning_text = state.get("project_context_warning", "")
        if warning_messages:
            extra_warning = (
                "Deep scan did not complete for all requested topics. "
                "Answer conservatively using the available project context or general retrieval only. "
                + "; ".join(warning_messages)
            )
            warning_text = f"{warning_text} {extra_warning}".strip()
        return {
            "deep_project_contexts": deep_contexts,
            "project_context_warning": warning_text,
        }

    def _scan_project_topic(
        self,
        *,
        active_project_path: str,
        topic: str,
        question: str,
        project_context: ProjectContext | None,
    ) -> tuple[bool, str, ProjectDeepContext | None]:
        provider = self._settings.project_scan_provider
        if provider in {"auto", "subagent"}:
            success, reason, deep_context, _ = self._project_subagent_scan_workflow.run(
                project_path=active_project_path,
                topic=topic,
                question=question,
                project_context=project_context,
            )
            if success:
                return True, reason, deep_context
            if provider == "subagent":
                fallback_success, fallback_reason, fallback_context, _ = self._project_deep_scan_workflow.run(
                    project_path=active_project_path,
                    topic=topic,
                    question=question,
                    provider="local",
                )
                if fallback_success:
                    return True, f"{reason}; fallback={fallback_reason}", fallback_context
                return False, f"{reason}; fallback={fallback_reason}", None

        success, reason, deep_context, _ = self._project_deep_scan_workflow.run(
            project_path=active_project_path,
            topic=topic,
            question=question,
            provider="local",
        )
        return success, reason, deep_context

    def _load_project_answer_memory_node(self, state: AnswerState) -> dict[str, object]:
        active_project_path = state.get("active_project_path")
        if not active_project_path:
            return {"project_answer_memory": {}}
        project_fingerprint = self._project_deep_scan_workflow.project_fingerprint(Path(active_project_path))
        memory_by_topic: dict[str, list[ProjectAnswerMemoryRecord]] = {}
        for topic in sorted(set(state.get("project_question_topics", {}).values())):
            records = self._project_answer_memory_store.load_recent_records(
                project_path=active_project_path,
                project_fingerprint=project_fingerprint,
                topic=topic,
                limit=3,
            )
            if records:
                memory_by_topic[topic] = records
        return {"project_answer_memory": memory_by_topic}

    def _retrieve_node(self, state: AnswerState) -> dict[str, object]:
        should_force_general_retrieval = (
            state.get("needs_project_context", False)
            and state.get("project_context") is None
            and not state.get("deep_project_contexts")
        )
        if not state.get("needs_general_retrieval", True) and not should_force_general_retrieval:
            return {"retrieved_context": {}}
        retrieval_mode = self._state_store.get_retrieval_mode(self._settings.retrieval_mode)
        retriever = QuestionRetriever(self._settings, self._vector_store, retrieval_mode=retrieval_mode)
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
                    "You generate interview answers from normalized questions. Use retrieved historical context only when it is relevant, and do not fabricate source usage. For questions about the current project, answer primarily from general technical principles, architecture patterns, and tradeoffs; use the current project only as a concise supporting example or proof point. Avoid over-indexing on file names, class names, or implementation trivia unless the user explicitly asks for exact code paths. If project context is unavailable, say so briefly and answer at a general engineering level instead of inventing details. If a question is algorithmic, include a Python reference implementation in the code field. For non-algorithm questions, leave the code field empty.",
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
                    "project_question_flags: {project_question_flags}\n"
                    "project_question_topics: {project_question_topics}\n"
                    "project_context: {project_context}\n"
                    "deep_project_contexts: {deep_project_contexts}\n"
                    "project_answer_memory: {project_answer_memory}\n"
                    "project_context_warning: {project_context_warning}\n"
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
                    project_question_flags=state.get("project_question_flags", {}),
                    project_question_topics=state.get("project_question_topics", {}),
                    project_context=state.get("project_context"),
                    deep_project_contexts=state.get("deep_project_contexts", {}),
                    project_answer_memory=state.get("project_answer_memory", {}),
                    project_context_warning=state.get("project_context_warning", ""),
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

    def _build_quick_answer_set(self, normalized_note: NormalizedNote) -> GeneratedAnswerSet:
        return GeneratedAnswerSet(
            note_id=normalized_note.note_id,
            note_url=normalized_note.note_url,
            title=f"{normalized_note.title}（快速版）" if normalized_note.title else normalized_note.note_id,
            answers=[self._build_quick_answer(question) for question in normalized_note.questions],
        )

    def _build_quick_answer(self, question: InterviewQuestion) -> GeneratedAnswerItem:
        answer_direction = self._quick_answer_direction(question)
        return GeneratedAnswerItem(
            question=question.question,
            short_answer=answer_direction,
            long_answer=(
                "这是快速版答案，用于先回发和确认题目已解析。"
                "详细版答案会在后台继续补全，完成后覆盖本地 answer.md / answers.json。"
            ),
            code=self._quick_code_hint(question),
            source_ids=[],
        )

    def _quick_answer_direction(self, question: InterviewQuestion) -> str:
        category = question.category.lower()
        if self._is_coding_question(question):
            return "先明确输入输出和边界条件，再讲解析/建模思路，最后补充时间复杂度和异常处理。"
        if "system" in category or "设计" in question.category or "agent" in question.question.lower():
            return "按整体架构、关键模块、状态流转、失败恢复和工程取舍来回答。"
        if "algorithm" in category or "算法" in question.category:
            return "先讲核心原理，再结合公式、复杂度、适用场景和工程优化回答。"
        if "backend" in category or "后端" in question.category:
            return "先讲机制原理，再讲排查路径、线上经验和落地取舍。"
        return "先给结论，再结合项目经历展开背景、做法、结果和复盘。"

    def _quick_code_hint(self, question: InterviewQuestion) -> str:
        if not self._is_coding_question(question):
            return ""
        return (
            "def solve(data, path):\n"
            "    # 快速版：详细实现会在后台补全答案中生成\n"
            "    raise NotImplementedError\n"
        )

    def _is_coding_question(self, question: InterviewQuestion) -> bool:
        lowered = question.question.lower()
        coding_markers = ["手撕", "编程题", "路径查询器", "写代码", "实现一个", "实现一个简易"]
        return any(marker in lowered for marker in coding_markers)

    def _is_project_related_question(self, question_text: str) -> bool:
        lowered = question_text.lower()
        project_markers = [
            "当前项目",
            "本项目",
            "这个项目",
            "该项目",
            "项目里",
            "项目中",
            "当前仓库",
            "这个仓库",
            "this project",
            "current project",
            "this repo",
            "current repo",
        ]
        return any(marker in lowered for marker in project_markers)

    def _detect_project_topic(self, question_text: str) -> str:
        lowered = question_text.lower()
        if any(marker in lowered for marker in ["memory", "记忆", "状态", "上下文"]):
            return "memory"
        if any(marker in lowered for marker in ["检索", "rag", "召回", "vector", "bm25", "hybrid"]):
            return "retrieval"
        if any(marker in lowered for marker in ["worker", "daemon", "后台", "轮询", "tmux"]):
            return "worker"
        if any(marker in lowered for marker in ["存储", "sqlite", "db", "数据库", "artifact"]):
            return "storage"
        if any(marker in lowered for marker in ["架构", "流程", "编排", "workflow", "graph", "agent"]):
            return "architecture"
        return "general"

    def _needs_deep_project_scan(self, question: InterviewQuestion) -> bool:
        lowered = question.question.lower()
        detail_markers = [
            "怎么做",
            "为什么",
            "具体",
            "实现",
            "链路",
            "流程",
            "模块",
            "区别",
            "tradeoff",
            "how",
            "why",
            "detail",
        ]
        topic = self._detect_project_topic(question.question)
        return topic != "general" or any(marker in lowered for marker in detail_markers)

    def _format_project_context(self, project_context: ProjectContext | None) -> str:
        if project_context is None:
            return ""
        return json.dumps(project_context.model_dump(mode="json"), ensure_ascii=False)

    def _format_deep_project_contexts(
        self,
        deep_project_contexts: dict[str, ProjectDeepContext],
    ) -> str:
        if not deep_project_contexts:
            return ""
        return json.dumps(
            {
                topic: context.model_dump(mode="json")
                for topic, context in deep_project_contexts.items()
            },
            ensure_ascii=False,
        )

    def _format_project_answer_memory(
        self,
        project_answer_memory: dict[str, list[ProjectAnswerMemoryRecord]],
    ) -> str:
        if not project_answer_memory:
            return ""
        return json.dumps(
            {
                topic: [record.model_dump(mode="json") for record in records]
                for topic, records in project_answer_memory.items()
            },
            ensure_ascii=False,
        )

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
        project_question_flags: dict[str, bool],
        project_question_topics: dict[str, str],
        project_context: ProjectContext | None,
        deep_project_contexts: dict[str, ProjectDeepContext],
        project_answer_memory: dict[str, list[ProjectAnswerMemoryRecord]],
        project_context_warning: str,
    ) -> list:
        try:
            batch_has_project_questions = any(
                project_question_flags.get(question.question, False) for question in questions
            )
            batch_question_topics = {
                question.question: project_question_topics.get(question.question, "general")
                for question in questions
                if project_question_flags.get(question.question, False)
            }
            batch_topics = set(batch_question_topics.values())
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
                    "project_question_flags": json.dumps(
                        {
                            question.question: project_question_flags.get(question.question, False)
                            for question in questions
                        },
                        ensure_ascii=False,
                    ),
                    "project_question_topics": json.dumps(batch_question_topics, ensure_ascii=False),
                    "project_context": self._format_project_context(
                        project_context if batch_has_project_questions else None
                    ),
                    "deep_project_contexts": self._format_deep_project_contexts(
                        {
                            topic: context
                            for topic, context in deep_project_contexts.items()
                            if topic in batch_topics
                        }
                    ),
                    "project_answer_memory": self._format_project_answer_memory(
                        {
                            topic: records
                            for topic, records in project_answer_memory.items()
                            if topic in batch_topics
                        }
                    ),
                    "project_context_warning": project_context_warning if batch_has_project_questions else "",
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
                        project_question_flags=project_question_flags,
                        project_question_topics=project_question_topics,
                        project_context=project_context,
                        deep_project_contexts=deep_project_contexts,
                        project_answer_memory=project_answer_memory,
                        project_context_warning=project_context_warning,
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
        try:
            self._persist_project_answer_memory(state=state, answer_set=answer_set)
        except Exception as exc:
            print(f"[project-answer-memory] note_id={state['note_id']} failed={exc}")
        return {"answer_path": str(answer_path), "markdown_path": str(markdown_path)}

    def _persist_project_answer_memory(
        self,
        *,
        state: AnswerState,
        answer_set: GeneratedAnswerSet,
    ) -> None:
        active_project_path = state.get("active_project_path")
        if not active_project_path:
            return
        project_context = state.get("project_context")
        deep_project_contexts = state.get("deep_project_contexts", {})
        project_question_flags = state.get("project_question_flags", {})
        project_question_topics = state.get("project_question_topics", {})
        fingerprint = self._project_deep_scan_workflow.project_fingerprint(
            Path(active_project_path)
        )
        for answer in answer_set.answers:
            if not project_question_flags.get(answer.question, False):
                continue
            topic = project_question_topics.get(answer.question, "general")
            key_files: list[str] = []
            if project_context is not None:
                key_files.extend(project_context.key_files)
            deep_context = deep_project_contexts.get(topic)
            if deep_context is not None:
                key_files.extend(deep_context.key_files)
            self._project_answer_memory_store.append_record(
                ProjectAnswerMemoryRecord(
                    project_path=active_project_path,
                    project_fingerprint=fingerprint,
                    note_id=answer_set.note_id,
                    question=answer.question,
                    topic=topic,
                    answer=answer.long_answer,
                    used_project_context=project_context is not None,
                    used_deep_scan=deep_context is not None,
                    key_files=self._dedupe_key_files(key_files),
                    created_at=datetime.now(UTC).isoformat(),
                )
            )

    def _dedupe_key_files(self, key_files: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for key_file in key_files:
            normalized = key_file.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

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
