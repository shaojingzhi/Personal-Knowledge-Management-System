from __future__ import annotations

import json
from importlib import import_module
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.telegram_state_store import TelegramStateStore
from xhs_interview_answer_copilot.storage.vector_store import IndexedQuestion, QuestionVectorStore
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.llm_retry import invoke_with_retry, is_budget_or_quota_error
from xhs_interview_answer_copilot.workflows.openai_clients import (
    build_chat_model,
    build_fallback_chat_model,
    fallback_available,
    fallback_model_name,
)
from xhs_interview_answer_copilot.workflows.retrieve_questions import QuestionRetriever


class ReactDecision(BaseModel):
    thought: str = Field(description="Reasoning summary for the next action.")
    action: Literal["search_knowledge", "answer_question"] = Field(
        description="Tool to call next."
    )
    action_input: str = Field(description="Input passed to the selected tool.")


class ReactAgentState(TypedDict):
    question: str
    steps: list[dict[str, str]]
    retrieved_context: NotRequired[list[IndexedQuestion]]
    final_answer: NotRequired[str]


class ReactAgentDemoWorkflow:
    def __init__(self, settings: Settings, vector_store: QuestionVectorStore) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._state_store = TelegramStateStore(sqlite_path=settings.sqlite_path)

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
            graph.add_node("answer_question", self._answer_question_node)
            graph.add_edge(START, "decide")
            graph.add_conditional_edges(
                "decide",
                self._route_after_decision,
                {
                    "search_knowledge": "search_knowledge",
                    "answer_question": "answer_question",
                },
            )
            graph.add_edge("search_knowledge", "decide")
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
                    "You are the Planning node in a minimal ReAct agent demo. Choose exactly one next tool. Use search_knowledge when external memory or prior examples would improve the answer. Use answer_question when enough context has already been gathered. Avoid repeated search_knowledge calls after an observation exists.",
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
        if self._has_search_observation(state) and decision.action == "search_knowledge":
            return ReactDecision(
                thought="A search observation already exists, so answer now.",
                action="answer_question",
                action_input=state["question"],
            )
        return decision

    def _fallback_decision(self, state: ReactAgentState) -> ReactDecision:
        if self._has_search_observation(state):
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

    def _has_search_observation(self, state: ReactAgentState) -> bool:
        return any(step.get("tool") == "search_knowledge" for step in state.get("steps", []))

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

    def _answer_question_node(self, state: ReactAgentState) -> dict[str, object]:
        question = state["steps"][-1]["action_input"]
        answer = self._answer_question(question=question, context=state.get("retrieved_context", []))
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

    def _answer_question(self, *, question: str, context: list[IndexedQuestion]) -> str:
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
                    "You are the answer_question tool in a small ReAct agent demo. Answer from general interview reasoning first, and use retrieved knowledge only as supporting context. Be concise but interview-ready.",
                ),
                (
                    "human",
                    "question: {question}\nretrieved_knowledge: {retrieved_knowledge}",
                ),
            ]
        )
        payload = {
            "question": question,
            "retrieved_knowledge": self._format_retrieved_context(context),
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
