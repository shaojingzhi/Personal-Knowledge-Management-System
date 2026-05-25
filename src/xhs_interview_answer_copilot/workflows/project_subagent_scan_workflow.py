from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

from pydantic import BaseModel, Field

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.project_deep_context_store import (
    ProjectDeepContextStore,
)
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.llm_retry import invoke_with_retry, is_budget_or_quota_error
from xhs_interview_answer_copilot.workflows.openai_clients import (
    build_chat_model,
    build_fallback_chat_model,
    fallback_available,
    fallback_model_name,
)
from xhs_interview_answer_copilot.workflows.project_deep_scan_workflow import (
    ProjectDeepScanWorkflow,
)
from xhs_interview_answer_copilot.workflows.schemas import ProjectDeepContext, ProjectContext


class _SubagentProjectScanOutput(BaseModel):
    confidence: str = Field(description="high, medium, or low confidence.")
    summary: str = Field(description="Topic-specific implementation summary for the repository.")
    key_findings: list[str] = Field(default_factory=list, description="Most important verified findings.")
    key_files: list[str] = Field(default_factory=list, description="Repository-relative files most relevant to the answer.")
    code_snippets: list[str] = Field(default_factory=list, description="Small code or config snippets supporting the answer.")
    followup_gaps: list[str] = Field(default_factory=list, description="Any uncertainty or missing verification gaps.")


class ProjectSubagentScanWorkflow:
    def __init__(
        self,
        settings: Settings,
        project_deep_context_store: ProjectDeepContextStore,
        local_scan_workflow: ProjectDeepScanWorkflow,
    ) -> None:
        self._settings = settings
        self._project_deep_context_store = project_deep_context_store
        self._local_scan_workflow = local_scan_workflow

    def run(
        self,
        *,
        project_path: str,
        topic: str,
        question: str,
        project_context: ProjectContext | None,
        force_refresh: bool = False,
    ) -> tuple[bool, str, ProjectDeepContext | None, str | None]:
        root = Path(project_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            return False, f"Invalid project path: {root}", None, None
        normalized_topic = self._local_scan_workflow.normalize_topic(topic)
        fingerprint = self._local_scan_workflow.project_fingerprint(root)
        question_hash = self._question_hash(question)
        if not force_refresh:
            cached = self._project_deep_context_store.load_context(
                project_path=str(root),
                topic=normalized_topic,
                fingerprint=fingerprint,
                provider="subagent",
                question_hash=question_hash,
            )
            if cached is not None:
                return (
                    True,
                    "Loaded cached subagent project context.",
                    cached,
                    str(
                        self._project_deep_context_store.context_path_for(
                            project_path=str(root),
                            topic=normalized_topic,
                            provider="subagent",
                            question_hash=question_hash,
                        )
                    ),
                )

        scored_files = self._local_scan_workflow.score_files(root=root, topic=normalized_topic)
        selected = scored_files[:8]
        evidence_payload = []
        for file_path, score in selected:
            evidence_payload.append(
                {
                    "path": str(file_path.relative_to(root)),
                    "score": score,
                    "snippets": self._local_scan_workflow.extract_snippets(
                        file_path=file_path,
                        root=root,
                        topic=normalized_topic,
                    )[:2],
                }
            )

        try:
            scan_output = self._invoke_subagent_model(
                project_name=root.name,
                project_path=str(root),
                topic=normalized_topic,
                question=question,
                project_context=project_context,
                evidence_payload=evidence_payload,
            )
            deep_context = ProjectDeepContext(
                project_name=root.name,
                project_path=str(root),
                topic=normalized_topic,
                fingerprint=fingerprint,
                scan_provider="subagent",
                confidence=scan_output.confidence.strip().lower() or "medium",
                summary=scan_output.summary,
                key_findings=scan_output.key_findings,
                key_files=scan_output.key_files,
                code_snippets=scan_output.code_snippets,
                followup_gaps=scan_output.followup_gaps,
            )
            context_path = self._project_deep_context_store.save_context(
                deep_context,
                provider="subagent",
                question_hash=question_hash,
            )
            return True, "Subagent project context refreshed.", deep_context, str(context_path)
        except Exception as exc:
            return False, f"Subagent project scan failed: {exc}", None, None

    def _invoke_subagent_model(
        self,
        *,
        project_name: str,
        project_path: str,
        topic: str,
        question: str,
        project_context: ProjectContext | None,
        evidence_payload: list[dict[str, object]],
    ) -> _SubagentProjectScanOutput:
        output_parsers_module = import_module("langchain_core.output_parsers")
        prompts_module = import_module("langchain_core.prompts")
        ChatPromptTemplate = prompts_module.ChatPromptTemplate
        PydanticOutputParser = output_parsers_module.PydanticOutputParser

        llm = build_chat_model(
            settings=self._settings,
            model_name=self._settings.project_scan_model,
            temperature=0,
        )
        parser = PydanticOutputParser(pydantic_object=_SubagentProjectScanOutput)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a repository scanning subagent. Synthesize implementation evidence for one topic-specific question about the active project. Use only the provided lightweight project context and repository evidence. Do not invent files, classes, or behavior that are not supported by the evidence. Frame findings in terms of general technical concepts, architecture patterns, and engineering tradeoffs first, then cite current-project evidence as concise examples. Avoid turning the output into a long file-by-file inventory unless exact code paths are necessary. Do not infer product behavior from the scanner's own internal implementation, directory skip lists, or meta-planning files unless the question is explicitly about the scanner itself.",
                ),
                (
                    "human",
                    "Return JSON only.\n"
                    "{format_instructions}\n"
                    "project_name: {project_name}\n"
                    "project_path: {project_path}\n"
                    "topic: {topic}\n"
                    "question: {question}\n"
                    "lightweight_project_context: {project_context}\n"
                    "repository_evidence: {repository_evidence}",
                ),
            ]
        )
        chain = prompt | llm
        payload = {
            "format_instructions": parser.get_format_instructions(),
            "project_name": project_name,
            "project_path": project_path,
            "topic": topic,
            "question": question,
            "project_context": "" if project_context is None else json.dumps(project_context.model_dump(mode="json"), ensure_ascii=False),
            "repository_evidence": json.dumps(evidence_payload, ensure_ascii=False),
        }
        try:
            response = invoke_with_retry(chain, payload)
        except Exception as exc:
            if not (fallback_available(self._settings) and is_budget_or_quota_error(exc)):
                raise
            fallback_llm = build_fallback_chat_model(
                settings=self._settings,
                model_name=fallback_model_name(self._settings, self._settings.project_scan_model),
                temperature=0,
            )
            response = invoke_with_retry(prompt | fallback_llm, payload)
        return parse_pydantic_response(parser, response)

    def _question_hash(self, question: str) -> str | None:
        normalized = question.strip()
        if not normalized:
            return None
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
