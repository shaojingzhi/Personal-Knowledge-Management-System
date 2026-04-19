from __future__ import annotations

from importlib import import_module
import re
from typing import NotRequired, TypedDict, cast

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
from xhs_interview_answer_copilot.workflows.llm_retry import invoke_with_retry
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.media_text_extractor import MediaTextExtractor
from xhs_interview_answer_copilot.workflows.openai_clients import build_chat_model
from xhs_interview_answer_copilot.workflows.schemas import (
    InterviewQuestion,
    NormalizedNote,
    SourceBundle,
    SourceLink,
)


class NormalizeState(TypedDict):
    note_id: str
    raw_note: dict[str, object]
    normalized_note: NotRequired[NormalizedNote]
    normalized_path: NotRequired[str]


class NormalizeNoteWorkflow:
    def __init__(
        self,
        settings: Settings,
        raw_store: RawArtifactStore,
        normalized_store: NormalizedArtifactStore,
    ) -> None:
        self._settings = settings
        self._raw_store = raw_store
        self._normalized_store = normalized_store

    def run(self, note_id: str) -> tuple[bool, str, str | None]:
        raw_note = self._raw_store.load_raw_payload(note_id)
        if raw_note is None:
            return False, f"Raw bundle not found for id: {note_id}", None
        if self._settings.openai_api_key is None and self._settings.openai_base_url is None:
            return False, "Configure OPENAI_API_KEY or OPENAI_BASE_URL first.", None

        try:
            graph_module = import_module("langgraph.graph")
            END = graph_module.END
            START = graph_module.START
            StateGraph = graph_module.StateGraph

            graph = StateGraph(NormalizeState)
            graph.add_node("normalize", self._normalize_node)
            graph.add_node("save", self._save_node)
            graph.add_edge(START, "normalize")
            graph.add_edge("normalize", "save")
            graph.add_edge("save", END)
            app = graph.compile()
            result = app.invoke({"note_id": note_id, "raw_note": raw_note})
        except Exception as exc:
            return False, f"Normalization failed: {exc}", None

        return True, "Normalization completed.", result.get("normalized_path")

    def _normalize_node(self, state: NormalizeState) -> dict[str, object]:
        output_parsers_module = import_module("langchain_core.output_parsers")
        prompts_module = import_module("langchain_core.prompts")
        ChatPromptTemplate = prompts_module.ChatPromptTemplate
        PydanticOutputParser = output_parsers_module.PydanticOutputParser

        llm = build_chat_model(
            settings=self._settings,
            model_name=self._settings.normalize_model,
            temperature=0,
        )
        parser = PydanticOutputParser(pydantic_object=NormalizedNote)
        raw_note = cast(dict[str, object], state["raw_note"])
        source_payload = self._build_source_payload(bundle_id=state["note_id"], raw_note=raw_note)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You normalize noisy interview source bundles into structured interview questions. Remove obvious UI noise and keep only useful interview content. Preserve the source language in the output. If the source content is mainly Chinese, the title, summary, tags, questions, categories, and keywords must stay in Chinese. Do not translate Chinese source material into English.",
                ),
                (
                    "human",
                    "Normalize this raw source bundle into structured interview questions.\n"
                    "Return JSON only.\n"
                    "preferred_output_language: {preferred_output_language}\n"
                    "{format_instructions}\n"
                    "source: {source}\n"
                    "note_id: {note_id}\n"
                    "note_url: {note_url}\n"
                    "title: {title}\n"
                    "body_text: {body_text}\n"
                    "asset_texts: {asset_texts}\n"
                    "image_urls: {image_urls}\n"
                    "asset_paths: {asset_paths}",
                ),
            ]
        )
        chain = prompt | llm
        response = invoke_with_retry(
            chain,
            {
                "format_instructions": parser.get_format_instructions(),
                "preferred_output_language": source_payload["preferred_output_language"],
                "source": source_payload["source"],
                "note_id": source_payload["note_id"],
                "note_url": source_payload["note_url"],
                "title": source_payload["title"],
                "body_text": source_payload["body_text"],
                "asset_texts": source_payload["asset_texts"],
                "image_urls": source_payload["image_urls"],
                "asset_paths": source_payload["asset_paths"],
            },
        )
        preferred_output_language = str(source_payload["preferred_output_language"])
        try:
            normalized_note = parse_pydantic_response(parser, response)
            if self._should_force_localized_fallback(
                normalized_note=normalized_note,
                preferred_output_language=preferred_output_language,
            ):
                normalized_note = self._fallback_normalize_from_source(
                    source_payload,
                    preferred_output_language=preferred_output_language,
                )
        except Exception:
            normalized_note = self._fallback_normalize_from_source(
                source_payload,
                preferred_output_language=preferred_output_language,
            )
        normalized_note.note_id = str(source_payload["note_id"])
        normalized_note.note_url = str(source_payload["note_url"])
        if not normalized_note.title:
            normalized_note.title = str(source_payload["title"])
        existing_warnings = list(normalized_note.warnings)
        normalized_note.warnings = existing_warnings + cast(
            list[str], source_payload.get("warnings", [])
        )
        return {"normalized_note": normalized_note}

    def _save_node(self, state: NormalizeState) -> dict[str, str]:
        normalized_note = cast(NormalizedNote, state.get("normalized_note"))
        normalized_path = self._normalized_store.save_normalized_note(
            note_id=state["note_id"],
            payload=normalized_note.model_dump(mode="json"),
        )
        return {"normalized_path": str(normalized_path)}

    def _build_source_payload(
        self,
        bundle_id: str,
        raw_note: dict[str, object],
    ) -> dict[str, object]:
        source_bundle = self._parse_source_bundle(bundle_id=bundle_id, raw_note=raw_note)
        first_link = source_bundle.links[0].url if source_bundle.links else ""
        asset_texts, warnings = MediaTextExtractor(self._settings).extract_from_paths(source_bundle.asset_paths)
        combined_text = self._join_non_empty(
            [source_bundle.title, "\n\n".join(source_bundle.text_blocks), "\n\n".join(asset_texts)]
        )
        return {
            "source": source_bundle.source,
            "note_id": source_bundle.bundle_id,
            "note_url": source_bundle.canonical_url or first_link,
            "title": source_bundle.title,
            "body_text": "\n\n".join(source_bundle.text_blocks),
            "asset_texts": asset_texts,
            "warnings": warnings,
            "image_urls": source_bundle.image_urls,
            "asset_paths": source_bundle.asset_paths,
            "preferred_output_language": self._detect_preferred_output_language(
                source_bundle=source_bundle,
                combined_text=combined_text,
            ),
        }

    def _detect_preferred_output_language(
        self,
        *,
        source_bundle: SourceBundle,
        combined_text: str,
    ) -> str:
        metadata = cast(dict[str, object], source_bundle.metadata)
        raw_message = cast(dict[str, object], metadata.get("raw_message", {}))
        sender = cast(dict[str, object], raw_message.get("from", {}))
        language_code = str(sender.get("language_code", ""))
        if language_code.lower().startswith("zh"):
            return "zh-CN"
        chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", combined_text))
        latin_words = len(re.findall(r"[A-Za-z]{2,}", combined_text))
        if chinese_characters >= max(8, latin_words):
            return "zh-CN"
        return "same-as-source"

    def _should_force_localized_fallback(
        self,
        *,
        normalized_note: NormalizedNote,
        preferred_output_language: str,
    ) -> bool:
        if preferred_output_language != "zh-CN":
            return False
        sample_text = "\n".join(
            [
                normalized_note.title,
                normalized_note.summary,
                *normalized_note.tags,
                *(question.question for question in normalized_note.questions),
                *(question.category for question in normalized_note.questions),
                *(keyword for question in normalized_note.questions for keyword in question.keywords),
            ]
        )
        chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", sample_text))
        latin_words = len(re.findall(r"[A-Za-z]{2,}", sample_text))
        return chinese_characters < max(6, latin_words)

    def _parse_source_bundle(
        self,
        bundle_id: str,
        raw_note: dict[str, object],
    ) -> SourceBundle:
        if "text_blocks" in raw_note and "source_type" in raw_note:
            return SourceBundle.model_validate(raw_note)

        source = str(raw_note.get("source", "xhs"))
        if source == "telegram":
            text = self._join_non_empty([raw_note.get("text"), raw_note.get("caption")])
            note_url = self._extract_telegram_url(raw_note=raw_note, text=text)
            return SourceBundle(
                bundle_id=str(raw_note.get("bundle_id", bundle_id)),
                source="telegram",
                source_type="message",
                canonical_url=note_url,
                title=self._build_title_from_text(text),
                text_blocks=[text] if text else [],
                links=[SourceLink(url=note_url, label="")] if note_url else [],
                asset_paths=cast(list[str], raw_note.get("asset_paths", [])),
                image_urls=[],
                metadata={
                    "entities": raw_note.get("entities", []),
                    "caption_entities": raw_note.get("caption_entities", []),
                    "raw_message": raw_note.get("raw_message", {}),
                },
            )

        note_url = str(raw_note.get("note_url", ""))
        body_text = str(raw_note.get("body_text", ""))
        return SourceBundle(
            bundle_id=str(raw_note.get("note_id", bundle_id)),
            source=source,
            source_type="note",
            canonical_url=note_url,
            title=str(raw_note.get("title", "")),
            text_blocks=[body_text] if body_text else [],
            links=[SourceLink(url=note_url, label="")] if note_url else [],
            asset_paths=cast(list[str], raw_note.get("asset_paths", [])),
            image_urls=cast(list[str], raw_note.get("image_urls", [])),
            metadata={},
        )

    def _join_non_empty(self, values: list[object]) -> str:
        parts: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped:
                parts.append(stripped)
        return "\n\n".join(parts)

    def _build_title_from_text(self, text: str) -> str:
        if not text:
            return ""
        first_line = text.splitlines()[0].strip()
        return first_line[:80]

    def _extract_first_url(self, text: str) -> str:
        for token in text.split():
            if token.startswith("http://") or token.startswith("https://"):
                return token.rstrip(").,!?]}>\"'"
                )
        return ""

    def _extract_telegram_url(self, raw_note: dict[str, object], text: str) -> str:
        entity_sources = [
            raw_note.get("entities", []),
            raw_note.get("caption_entities", []),
        ]
        for entity_list in entity_sources:
            if not isinstance(entity_list, list):
                continue
            for entity in entity_list:
                if not isinstance(entity, dict):
                    continue
                if entity.get("type") == "text_link" and isinstance(entity.get("url"), str):
                    return str(entity["url"])
        return self._extract_first_url(text)

    def _fallback_normalize_from_source(
        self,
        source_payload: dict[str, object],
        *,
        preferred_output_language: str,
    ) -> NormalizedNote:
        asset_texts = cast(list[object], source_payload.get("asset_texts", []))
        asset_text = "\n\n".join(text for text in asset_texts if isinstance(text, str))
        combined_text = self._join_non_empty(
            [source_payload.get("title"), source_payload.get("body_text"), asset_text]
        )
        questions = self._extract_questions_from_text(
            combined_text,
            preferred_output_language=preferred_output_language,
        )
        title = str(source_payload.get("title", "")).strip() or self._derive_title_from_text(combined_text)
        summary = self._derive_summary_from_text(
            title=title,
            text=combined_text,
            question_count=len(questions),
            preferred_output_language=preferred_output_language,
        )
        warnings = [
            "Normalization used local fallback extraction because the model response was empty or invalid."
        ]
        return NormalizedNote(
            note_id=str(source_payload.get("note_id", "")),
            note_url=str(source_payload.get("note_url", "")),
            title=title,
            summary=summary,
            tags=self._derive_tags_from_text(
                title=title,
                text=combined_text,
                preferred_output_language=preferred_output_language,
            ),
            warnings=warnings,
            questions=questions,
        )

    def _extract_questions_from_text(
        self,
        text: str,
        *,
        preferred_output_language: str,
    ) -> list[InterviewQuestion]:
        questions: list[InterviewQuestion] = []
        seen: set[str] = set()
        current_parts: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self._is_noise_line(line):
                continue
            if self._starts_numbered_item(line):
                if current_parts:
                    question = self._build_question(
                        " ".join(current_parts),
                        preferred_output_language=preferred_output_language,
                    )
                    if question is not None and question.question not in seen:
                        questions.append(question)
                        seen.add(question.question)
                current_parts = [self._strip_number_prefix(line)]
                continue
            if current_parts:
                current_parts.append(line)
        if current_parts:
            question = self._build_question(
                " ".join(current_parts),
                preferred_output_language=preferred_output_language,
            )
            if question is not None and question.question not in seen:
                questions.append(question)
        return questions

    def _build_question(
        self,
        text: str,
        *,
        preferred_output_language: str,
    ) -> InterviewQuestion | None:
        question = re.sub(r"\s+", " ", text).strip(" -•·。")
        if len(question) < 4:
            return None
        if not question.endswith(("?", "？")):
            question = f"{question}？"
        return InterviewQuestion(
            question=question,
            category=self._infer_question_category(
                question,
                preferred_output_language=preferred_output_language,
            ),
            keywords=self._extract_keywords(
                question,
                preferred_output_language=preferred_output_language,
            ),
        )

    def _derive_title_from_text(self, text: str) -> str:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or self._is_noise_line(line):
                continue
            if self._starts_numbered_item(line):
                continue
            return line[:80]
        return "Telegram OCR note"

    def _derive_summary_from_text(
        self,
        title: str,
        text: str,
        question_count: int,
        *,
        preferred_output_language: str,
    ) -> str:
        if preferred_output_language == "zh-CN":
            if title:
                return f"{title}，通过本地回退流程提取出 {question_count} 个面试问题。"
            return f"通过本地回退流程从源文本中提取出 {question_count} 个面试问题。"
        if title:
            return f"{title}. Extracted {question_count} interview questions via local fallback normalization."
        return f"Extracted {question_count} interview questions via local fallback normalization."

    def _derive_tags_from_text(
        self,
        title: str,
        text: str,
        *,
        preferred_output_language: str,
    ) -> list[str]:
        if preferred_output_language == "zh-CN":
            candidates = [
                ("智能体", ["agent", "智能体"]),
                ("检索增强", ["rag"]),
                ("Java", ["java"]),
                ("JVM", ["jvm", "垃圾回收", "gc"]),
                ("大模型", ["大模型", "ai coding", "llm"]),
                ("面经", ["凉经", "面经", "一面", "二面"]),
            ]
        else:
            candidates = [
                ("Agent", ["agent", "智能体"]),
                ("RAG", ["rag"]),
                ("Java", ["java"]),
                ("JVM", ["jvm", "垃圾回收", "gc"]),
                ("AIGC", ["大模型", "ai coding", "llm"]),
                ("Interview", ["凉经", "面经", "一面", "二面"]),
            ]
        haystack = f"{title}\n{text}".lower()
        tags = [label for label, needles in candidates if any(needle.lower() in haystack for needle in needles)]
        return tags[:8]

    def _infer_question_category(self, question: str, *, preferred_output_language: str) -> str:
        lowered = question.lower()
        if preferred_output_language == "zh-CN":
            if any(keyword in lowered for keyword in ["java", "jvm", "gc", "aop", "cglib", "cpu", "oom"]):
                return "后端"
            if any(keyword in lowered for keyword in ["agent", "rag", "意图识别", "知识库", "记忆", "大模型", "openai", "coding"]):
                return "系统设计"
            if any(keyword in lowered for keyword in ["自我介绍", "成就感", "团队", "挑战", "反问", "怎么看"]):
                return "行为"
            return "通用"
        if any(keyword in lowered for keyword in ["java", "jvm", "gc", "aop", "cglib", "cpu", "oom"]):
            return "backend"
        if any(keyword in lowered for keyword in ["agent", "rag", "意图识别", "知识库", "记忆", "大模型", "openai", "coding"]):
            return "system-design"
        if any(keyword in lowered for keyword in ["自我介绍", "成就感", "团队", "挑战", "反问", "怎么看"]):
            return "behavior"
        return "general"

    def _extract_keywords(self, question: str, *, preferred_output_language: str) -> list[str]:
        keywords: list[str] = []
        if preferred_output_language == "zh-CN":
            token_map = [
                ("智能体", "agent"),
                ("检索增强", "rag"),
                ("Java", "java"),
                ("JVM", "jvm"),
                ("AOP", "aop"),
                ("OOM", "oom"),
                ("CPU", "cpu"),
                ("意图识别", "意图识别"),
                ("记忆系统", "记忆"),
                ("AI 编码", "ai coding"),
            ]
        else:
            token_map = [
                ("Agent", "agent"),
                ("RAG", "rag"),
                ("Java", "java"),
                ("JVM", "jvm"),
                ("AOP", "aop"),
                ("OOM", "oom"),
                ("CPU", "cpu"),
                ("Intent Recognition", "意图识别"),
                ("Memory System", "记忆"),
                ("AI Coding", "ai coding"),
            ]
        for label, needle in token_map:
            if needle in question.lower() or label.lower() in question.lower():
                keywords.append(label)
        return keywords[:5]

    def _is_noise_line(self, line: str) -> bool:
        lowered = line.lower()
        if any(token in lowered for token in ["长按扫描", "二维码", "小红书", "点赞", "收藏", "分享"]):
            return True
        return bool(re.fullmatch(r"[\W_0-9①②③④⑤⑥⑦⑧⑨⑩=®©:：.-]+", line))

    def _starts_numbered_item(self, line: str) -> bool:
        return bool(re.match(r"^(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩])[.、]\s*", line))

    def _strip_number_prefix(self, line: str) -> str:
        return re.sub(r"^(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩])[.、]\s*", "", line).strip()
