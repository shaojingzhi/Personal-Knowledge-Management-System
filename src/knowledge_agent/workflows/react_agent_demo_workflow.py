from __future__ import annotations

import json
from importlib import import_module
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from knowledge_agent.config import Settings
from knowledge_agent.storage.telegram_state_store import TelegramStateStore
from knowledge_agent.storage.vector_store import IndexedQuestion, QuestionVectorStore
from knowledge_agent.workflows.json_output_parser import parse_pydantic_response
from knowledge_agent.workflows.llm_retry import invoke_with_retry, is_budget_or_quota_error
from knowledge_agent.workflows.openai_clients import (
    build_chat_model,
    build_fallback_chat_model,
    fallback_available,
    fallback_model_name,
)
from knowledge_agent.workflows.project_context_workflow import ProjectContextWorkflow
from knowledge_agent.workflows.project_deep_scan_workflow import ProjectDeepScanWorkflow
from knowledge_agent.workflows.project_subagent_scan_workflow import ProjectSubagentScanWorkflow
from knowledge_agent.workflows.retrieve_questions import QuestionRetriever
from knowledge_agent.workflows.schemas import ProjectContext, ProjectDeepContext


class ReactDecision(BaseModel):
    thought: str = Field(description="Reasoning summary for the next action.")
    action: Literal[
        "search_knowledge",
        "read_project_context",
        "scan_project_source",
        "answer_question",
    ] = Field(
        description="Tool to call next."
    )
    action_input: str = Field(description="Input passed to the selected tool.")


class ReactAgentState(TypedDict):
    question: str
    steps: list[dict[str, str]]
    retrieved_context: NotRequired[list[IndexedQuestion]]
    project_context: NotRequired[ProjectContext]
    deep_project_context: NotRequired[ProjectDeepContext]
    final_answer: NotRequired[str]


_PROJECT_TOPICS = ("memory", "retrieval", "worker", "storage", "architecture", "general")


class ReactAgentDemoWorkflow:
    def __init__(
        self,
        settings: Settings,
        vector_store: QuestionVectorStore,
        project_context_workflow: ProjectContextWorkflow,
        project_deep_scan_workflow: ProjectDeepScanWorkflow,
        project_subagent_scan_workflow: ProjectSubagentScanWorkflow,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)
        self._project_context_workflow = project_context_workflow
        self._project_deep_scan_workflow = project_deep_scan_workflow
        self._project_subagent_scan_workflow = project_subagent_scan_workflow

    def run(self, question: str) -> tuple[bool, str, str, list[dict[str, str]]]:
        if self._settings.openai_api_key is None and self._settings.openai_base_url is None:
            return False, "Configure OPENAI_API_KEY or OPENAI_BASE_URL first.", "", []
        try:
            graph_module = import_module("langgraph.graph")
            END = graph_module.END
            START = graph_module.START
            StateGraph = graph_module.StateGraph

            graph = StateGraph(ReactAgentState)
            graph.add_node("decide", self._decide_node)
            graph.add_node("search_knowledge", self._search_knowledge_node)
            graph.add_node("read_project_context", self._read_project_context_node)
            graph.add_node("scan_project_source", self._scan_project_source_node)
            graph.add_node("answer_question", self._answer_question_node)
            graph.add_edge(START, "decide")
            graph.add_conditional_edges(
                "decide",
                self._route_after_decision,
                {
                    "search_knowledge": "search_knowledge",
                    "read_project_context": "read_project_context",
                    "scan_project_source": "scan_project_source",
                    "answer_question": "answer_question",
                },
            )
            graph.add_edge("search_knowledge", "decide")
            graph.add_edge("read_project_context", "decide")
            graph.add_edge("scan_project_source", "decide")
            graph.add_edge("answer_question", END)
            app = graph.compile()
            result = app.invoke({"question": question, "steps": []})
        except Exception as exc:
            return False, f"ReAct agent demo failed: {exc}", "", []
        return True, "ReAct agent demo completed.", result.get("final_answer", ""), result.get("steps", [])

    def _decide_node(self, state: ReactAgentState) -> dict[str, object]:
        decision = self._llm_decide(state)
        steps = [*state.get("steps", []), decision.model_dump(mode="json")]
        return {"steps": steps}

    def _llm_decide(self, state: ReactAgentState) -> ReactDecision:
        output_parsers_module = import_module("langchain_core.output_parsers")
        prompts_module = import_module("langchain_core.prompts")
        ChatPromptTemplate = prompts_module.ChatPromptTemplate
        PydanticOutputParser = output_parsers_module.PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=ReactDecision)
        llm = build_chat_model(
            settings=self._settings,
            model_name=self._settings.answer_model,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the Planning node in a minimal ReAct agent demo. Choose exactly one next tool. Use search_knowledge when external memory or prior examples would improve the answer. Use read_project_context when the user asks about the current project, this project, this repo, or implementation details tied to the active repository. Use scan_project_source when lightweight project context is not enough and repository implementation evidence is needed for topics such as memory, retrieval, worker, storage, or architecture. Use answer_question when enough context has already been gathered. Avoid repeated calls to the same information-gathering tool after a useful observation already exists.",
                ),
                (
                    "human",
                    "Return JSON only.\n"
                    "{format_instructions}\n"
                    "question: {question}\n"
                    "steps_so_far: {steps_so_far}",
                ),
            ]
        )
        try:
            payload = {
                "format_instructions": parser.get_format_instructions(),
                "question": state["question"],
                "steps_so_far": json.dumps(state.get("steps", []), ensure_ascii=False),
            }
            try:
                response = invoke_with_retry(prompt | llm, payload)
            except Exception as exc:
                if not (fallback_available(self._settings) and is_budget_or_quota_error(exc)):
                    raise
                fallback_llm = build_fallback_chat_model(
                    settings=self._settings,
                    model_name=fallback_model_name(self._settings, self._settings.answer_model),
                    temperature=0,
                )
                response = invoke_with_retry(prompt | fallback_llm, payload)
            decision = parse_pydantic_response(parser, response)
        except Exception:
            decision = self._fallback_decision(state)
        if self._has_tool_observation(state, decision.action):
            return ReactDecision(
                thought="That tool already produced an observation, so answer now.",
                action="answer_question",
                action_input=state["question"],
            )
        return decision

    def _fallback_decision(self, state: ReactAgentState) -> ReactDecision:
        if self._is_project_question(state["question"]):
            if not self._has_tool_observation(state, "read_project_context"):
                return ReactDecision(
                    thought="Need the active project summary before answering a project-specific question.",
                    action="read_project_context",
                    action_input=state["question"],
                )
            if not self._has_tool_observation(state, "scan_project_source"):
                return ReactDecision(
                    thought="Need repository implementation evidence before answering the project-specific question.",
                    action="scan_project_source",
                    action_input=self._detect_project_topic(state["question"]),
                )
            return ReactDecision(
                thought="Project context has been gathered; answer the question now.",
                action="answer_question",
                action_input=state["question"],
            )
        if self._has_tool_observation(state, "search_knowledge"):
            return ReactDecision(
                thought="Knowledge has been gathered; answer the question now.",
                action="answer_question",
                action_input=state["question"],
            )
        return ReactDecision(
            thought="Need supporting knowledge before answering.",
            action="search_knowledge",
            action_input=state["question"],
        )

    def _has_tool_observation(self, state: ReactAgentState, tool_name: str) -> bool:
        return any(step.get("tool") == tool_name for step in state.get("steps", []))

    def _route_after_decision(self, state: ReactAgentState) -> str:
        last_step = state["steps"][-1]
        return last_step["action"]

    def _search_knowledge_node(self, state: ReactAgentState) -> dict[str, object]:
        query = state["steps"][-1]["action_input"]
        retrieval_mode = self._state_store.get_retrieval_mode(self._settings.retrieval_mode)
        retriever = QuestionRetriever(
            settings=self._settings,
            vector_store=self._vector_store,
            retrieval_mode=retrieval_mode,
        )
        success, reason, results = retriever.search(
            query=query,
            top_k=self._settings.retrieval_top_k,
        )
        observation = self._format_search_observation(success=success, reason=reason, results=results)
        return {
            "retrieved_context": results,
            "steps": [
                *state.get("steps", []),
                {
                    "tool": "search_knowledge",
                    "input": query,
                    "observation": observation,
                },
            ],
        }

    def _read_project_context_node(self, state: ReactAgentState) -> dict[str, object]:
        active_project_path = self._state_store.get_active_project_path()
        if not active_project_path:
            observation = json.dumps(
                {
                    "success": False,
                    "reason": "No active project is configured.",
                },
                ensure_ascii=False,
            )
            return {
                "steps": [
                    *state.get("steps", []),
                    {
                        "tool": "read_project_context",
                        "input": state["question"],
                        "observation": observation,
                    },
                ]
            }
        success, reason, project_context, _ = self._project_context_workflow.run(
            active_project_path,
            force_refresh=True,
        )
        observation = self._format_project_context_observation(
            success=success,
            reason=reason,
            project_context=project_context,
        )
        payload: dict[str, object] = {
            "steps": [
                *state.get("steps", []),
                {
                    "tool": "read_project_context",
                    "input": active_project_path,
                    "observation": observation,
                },
            ]
        }
        if success and project_context is not None:
            payload["project_context"] = project_context
        return payload

    def _scan_project_source_node(self, state: ReactAgentState) -> dict[str, object]:
        active_project_path = self._state_store.get_active_project_path()
        topic = self._normalize_scan_topic(state["steps"][-1]["action_input"])
        if not active_project_path:
            observation = json.dumps(
                {
                    "success": False,
                    "reason": "No active project is configured.",
                    "topic": topic,
                },
                ensure_ascii=False,
            )
            return {
                "steps": [
                    *state.get("steps", []),
                    {
                        "tool": "scan_project_source",
                        "input": topic,
                        "observation": observation,
                    },
                ]
            }
        success, reason, deep_context = self._scan_project_source(
            active_project_path=active_project_path,
            topic=topic,
            question=state["question"],
            project_context=state.get("project_context"),
        )
        observation = self._format_deep_project_context_observation(
            success=success,
            reason=reason,
            topic=topic,
            deep_context=deep_context,
        )
        payload: dict[str, object] = {
            "steps": [
                *state.get("steps", []),
                {
                    "tool": "scan_project_source",
                    "input": topic,
                    "observation": observation,
                },
            ]
        }
        if success and deep_context is not None:
            payload["deep_project_context"] = deep_context
        return payload

    def _answer_question_node(self, state: ReactAgentState) -> dict[str, object]:
        question = state["steps"][-1]["action_input"]
        answer = self._answer_question(
            question=question,
            context=state.get("retrieved_context", []),
            project_context=state.get("project_context"),
            deep_project_context=state.get("deep_project_context"),
        )
        return {
            "final_answer": answer,
            "steps": [
                *state.get("steps", []),
                {
                    "tool": "answer_question",
                    "input": question,
                    "observation": answer,
                },
            ],
        }

    def _answer_question(
        self,
        *,
        question: str,
        context: list[IndexedQuestion],
        project_context: ProjectContext | None,
        deep_project_context: ProjectDeepContext | None,
    ) -> str:
        prompts_module = import_module("langchain_core.prompts")
        ChatPromptTemplate = prompts_module.ChatPromptTemplate
        llm = build_chat_model(
            settings=self._settings,
            model_name=self._settings.answer_model,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the answer_question tool in a small ReAct agent demo. Answer from general interview reasoning first, and use retrieved knowledge only as supporting context. If project context or repository scan evidence exists, use it as concise supporting evidence for current-project questions. Be concise but interview-ready.",
                ),
                (
                    "human",
                    "question: {question}\nretrieved_knowledge: {retrieved_knowledge}\nproject_context: {project_context}\ndeep_project_context: {deep_project_context}",
                ),
            ]
        )
        payload = {
            "question": question,
            "retrieved_knowledge": self._format_retrieved_context(context),
            "project_context": self._format_project_context_for_prompt(project_context),
            "deep_project_context": self._format_deep_project_context_for_prompt(deep_project_context),
        }
        try:
            response = invoke_with_retry(prompt | llm, payload)
        except Exception as exc:
            if not (fallback_available(self._settings) and is_budget_or_quota_error(exc)):
                raise
            fallback_llm = build_fallback_chat_model(
                settings=self._settings,
                model_name=fallback_model_name(self._settings, self._settings.answer_model),
                temperature=0,
            )
            response = invoke_with_retry(prompt | fallback_llm, payload)
        content = getattr(response, "content", response)
        return str(content).strip()

    def _format_search_observation(
        self,
        *,
        success: bool,
        reason: str,
        results: list[IndexedQuestion],
    ) -> str:
        payload = {
            "success": success,
            "reason": reason,
            "matches": [
                {
                    "score": round(result.score, 4),
                    "note_id": result.note_id,
                    "question": result.question,
                    "summary": result.summary,
                }
                for result in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _format_retrieved_context(self, context: list[IndexedQuestion]) -> str:
        return json.dumps(
            [
                {
                    "question": item.question,
                    "summary": item.summary,
                    "score": round(item.score, 4),
                }
                for item in context
            ],
            ensure_ascii=False,
        )

    def _scan_project_source(
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

    def _format_project_context_observation(
        self,
        *,
        success: bool,
        reason: str,
        project_context: ProjectContext | None,
    ) -> str:
        payload = {
            "success": success,
            "reason": reason,
            "project_context": None
            if project_context is None
            else {
                "project_name": project_context.project_name,
                "summary": project_context.summary,
                "tech_stack": project_context.tech_stack,
                "key_files": project_context.key_files[:8],
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _format_deep_project_context_observation(
        self,
        *,
        success: bool,
        reason: str,
        topic: str,
        deep_context: ProjectDeepContext | None,
    ) -> str:
        payload = {
            "success": success,
            "reason": reason,
            "topic": topic,
            "deep_project_context": None
            if deep_context is None
            else {
                "summary": deep_context.summary,
                "key_findings": deep_context.key_findings[:6],
                "key_files": deep_context.key_files[:8],
                "confidence": deep_context.confidence,
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _format_project_context_for_prompt(self, project_context: ProjectContext | None) -> str:
        if project_context is None:
            return ""
        return json.dumps(project_context.model_dump(mode="json"), ensure_ascii=False)

    def _format_deep_project_context_for_prompt(
        self, deep_project_context: ProjectDeepContext | None
    ) -> str:
        if deep_project_context is None:
            return ""
        return json.dumps(deep_project_context.model_dump(mode="json"), ensure_ascii=False)

    def _is_project_question(self, question: str) -> bool:
        lowered = question.lower()
        project_markers = (
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
        )
        return any(marker in lowered for marker in project_markers)

    def _detect_project_topic(self, question: str) -> str:
        lowered = question.lower()
        if any(marker in lowered for marker in ("memory", "记忆", "状态", "上下文")):
            return "memory"
        if any(marker in lowered for marker in ("检索", "rag", "召回", "vector", "bm25", "hybrid")):
            return "retrieval"
        if any(marker in lowered for marker in ("worker", "daemon", "后台", "轮询", "tmux")):
            return "worker"
        if any(marker in lowered for marker in ("存储", "sqlite", "db", "数据库", "artifact")):
            return "storage"
        if any(marker in lowered for marker in ("架构", "流程", "编排", "workflow", "graph", "agent")):
            return "architecture"
        return "general"

    def _normalize_scan_topic(self, action_input: str) -> str:
        normalized = action_input.strip().lower()
        if normalized in _PROJECT_TOPICS:
            return normalized
        return self._detect_project_topic(action_input)
