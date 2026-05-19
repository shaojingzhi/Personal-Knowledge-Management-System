from __future__ import annotations

from importlib import import_module
from typing import NotRequired, TypedDict, cast

from xhs_interview_answer_copilot.config import Settings
from xhs_interview_answer_copilot.storage.normalized_artifact_store import (
    NormalizedArtifactStore,
)
from xhs_interview_answer_copilot.storage.raw_artifact_store import RawArtifactStore
from xhs_interview_answer_copilot.workflows.json_output_parser import parse_pydantic_response
from xhs_interview_answer_copilot.workflows.media_text_extractor import MediaTextExtractor
from xhs_interview_answer_copilot.workflows.openai_clients import build_chat_model
from xhs_interview_answer_copilot.workflows.schemas import (
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
                    "You normalize noisy interview source bundles into structured interview questions. Remove obvious UI noise and keep only useful interview content.",
                ),
                (
                    "human",
                    "Normalize this raw source bundle into structured interview questions.\n"
                    "Return JSON only.\n"
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
        response = chain.invoke(
            {
                "format_instructions": parser.get_format_instructions(),
                "source": source_payload["source"],
                "note_id": source_payload["note_id"],
                "note_url": source_payload["note_url"],
                "title": source_payload["title"],
                "body_text": source_payload["body_text"],
                "asset_texts": source_payload["asset_texts"],
                "image_urls": source_payload["image_urls"],
                "asset_paths": source_payload["asset_paths"],
            }
        )
        normalized_note = parse_pydantic_response(parser, response)
        normalized_note.note_id = source_payload["note_id"]
        normalized_note.note_url = source_payload["note_url"]
        if not normalized_note.title:
            normalized_note.title = source_payload["title"]
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
        }

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
