from __future__ import annotations

from pydantic import BaseModel, Field


class InterviewQuestion(BaseModel):
    question: str = Field(description="Normalized interview question text.")
    category: str = Field(
        description="Question category such as backend, algorithm, system-design, or behavior."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Short keywords for retrieval and later indexing.",
    )


class NormalizedNote(BaseModel):
    note_id: str = Field(description="Original Xiaohongshu note id.")
    note_url: str = Field(description="Original Xiaohongshu note url.")
    title: str = Field(description="Normalized note title if available.")
    summary: str = Field(description="One short summary of the interview post.")
    tags: list[str] = Field(default_factory=list, description="Compact tags for the post.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Best-effort extraction or normalization warnings.",
    )
    questions: list[InterviewQuestion] = Field(
        default_factory=list,
        description="Structured interview questions extracted from the post.",
    )


class SourceLink(BaseModel):
    url: str = Field(description="Original link extracted from the source bundle.")
    label: str = Field(default="", description="Optional visible text or label for the link.")


class SourceBundle(BaseModel):
    bundle_id: str = Field(description="Stable id for one ingested source bundle.")
    source: str = Field(description="Source system name such as telegram, feishu, or xhs-web.")
    source_type: str = Field(description="Bundle type such as message, note, webhook, or import.")
    canonical_url: str = Field(default="", description="Best-effort primary URL for the source bundle.")
    title: str = Field(default="", description="Best-effort source title.")
    text_blocks: list[str] = Field(
        default_factory=list,
        description="Ordered text snippets extracted from the source.",
    )
    links: list[SourceLink] = Field(
        default_factory=list,
        description="Links extracted from the source bundle.",
    )
    asset_paths: list[str] = Field(
        default_factory=list,
        description="Local downloaded asset paths for later OCR or multimodal reading.",
    )
    image_urls: list[str] = Field(
        default_factory=list,
        description="Remote image URLs retained when the source provides them.",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Provider-specific metadata retained for debugging and future adapters.",
    )


class RetrievedContext(BaseModel):
    source_id: str = Field(description="Stable id for one retrieved indexed question.")
    note_id: str = Field(description="Source note id for the retrieved question.")
    question: str = Field(description="Retrieved similar question text.")
    summary: str = Field(description="Short summary from the source note.")
    score: float = Field(description="Similarity score for retrieval ordering.")


class GeneratedAnswerItem(BaseModel):
    question: str = Field(description="The normalized target interview question.")
    short_answer: str = Field(description="One concise review-friendly answer.")
    long_answer: str = Field(description="A fuller interview-ready answer in 3-5 sentences.")
    code: str = Field(
        default="",
        description="Python reference implementation for algorithm questions; empty for non-algorithm questions.",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Retrieved source ids actually used for grounding.",
    )


class GeneratedAnswerSet(BaseModel):
    note_id: str = Field(description="Original Xiaohongshu note id.")
    note_url: str = Field(description="Original Xiaohongshu note url.")
    title: str = Field(description="Normalized note title.")
    answers: list[GeneratedAnswerItem] = Field(
        default_factory=list,
        description="Generated answer set for all extracted questions.",
    )


class ProjectContext(BaseModel):
    project_name: str = Field(description="Resolved project name.")
    project_path: str = Field(description="Absolute path to the active project root.")
    summary: str = Field(description="Short overview of what the project does.")
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Relevant technologies, frameworks, and infrastructure pieces.",
    )
    memory_system: str = Field(
        description="How the project persists memory, state, or reusable artifacts.",
    )
    retrieval_system: str = Field(
        description="How the project retrieves prior knowledge or indexed records.",
    )
    workflow_orchestration: str = Field(
        description="How the project coordinates its main execution flow.",
    )
    storage: str = Field(description="How the project stores durable state and artifacts.")
    background_jobs: str = Field(
        description="How the project handles long-running or background work.",
    )
    key_files: list[str] = Field(
        default_factory=list,
        description="Repository-relative files that best support the summary.",
    )


class ProjectDeepContext(BaseModel):
    project_name: str = Field(description="Resolved active project name.")
    project_path: str = Field(description="Absolute active project path.")
    topic: str = Field(description="Topic that triggered the deep scan, such as retrieval or memory.")
    fingerprint: str = Field(description="Project fingerprint used for cache invalidation.")
    scan_provider: str = Field(description="Which runtime scanner produced this context, such as subagent or local.")
    confidence: str = Field(description="Confidence label such as high, medium, or low.")
    summary: str = Field(description="Topic-specific repository summary from the deep scan.")
    key_findings: list[str] = Field(
        default_factory=list,
        description="Short findings the answer generator can reuse directly.",
    )
    key_files: list[str] = Field(
        default_factory=list,
        description="Repository-relative files most relevant to this topic.",
    )
    code_snippets: list[str] = Field(
        default_factory=list,
        description="Small code excerpts or references supporting the topic-specific summary.",
    )
    followup_gaps: list[str] = Field(
        default_factory=list,
        description="Known gaps or caveats that the scanner could not fully verify.",
    )


class ProjectAnswerMemoryRecord(BaseModel):
    project_path: str = Field(description="Absolute active project path used for the answer.")
    project_fingerprint: str = Field(description="Project fingerprint captured when the answer was created.")
    note_id: str = Field(description="Source note id that produced the answer.")
    question: str = Field(description="Original project-specific question.")
    topic: str = Field(description="Detected project topic for the question.")
    answer: str = Field(description="Final detailed answer text.")
    used_project_context: bool = Field(description="Whether cached project context was injected.")
    used_deep_scan: bool = Field(description="Whether a deep repo scan result was injected.")
    key_files: list[str] = Field(
        default_factory=list,
        description="Repository-relative files that supported the answer.",
    )
    created_at: str = Field(description="ISO timestamp when the answer memory record was stored.")
