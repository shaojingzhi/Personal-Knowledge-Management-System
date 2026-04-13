# Pending Items

## Current hardening items

- Add Telegram bot authentication and allowlist checks so only trusted senders can trigger ingestion.
- Make Telegram media download and retry handling resilient to network or API hiccups.
- Add clearer validation for mixed-input bundles containing links, captions, screenshots, and pasted text.
- Confirm and lock the primary input contract: Telegram should accept Xiaohongshu share images as the preferred practical input shape for text-heavy posts.
- Record explicit limitations for image-heavy posts whose main content spans multiple images or cannot be captured well by one share image.
- Replace whole-page note extraction with note-scoped selectors so navigation text, comments, and recommendations do not pollute `raw.json`.
- Replace the fixed wait in note extraction with better note-detail readiness checks.
- Fail extraction more clearly when title, body, and image results are all too sparse to be useful.
- Make note re-indexing atomic so a failed refresh does not temporarily delete all vectors for one note.
- Add stronger validation for generated `source_ids` so answers cannot claim grounding that was not actually retrieved for the matching question.
- Add model/version metadata to `raw.json`, `normalized.json`, and `answers.json` so reruns are easier to compare and debug.
- Make runtime config loading consistent with README so local `.env` values are picked up without manually sourcing them in shell commands.

## Near-term implementation items

- Implement Telegram message ingestion as the new primary entry point.
- Persist raw Telegram message bundles and attachments locally.
- Extend raw source handling to download or cache Telegram-delivered images in addition to storing `raw.json`.
- Implement OCR for single-share-image Xiaohongshu inputs so text content can flow into the existing normalization path.
- Add OCR or multimodal image text extraction so image-heavy posts are not normalized from URLs alone.
- Make Telegram media-first bundles usable by reading downloaded assets instead of only passing asset paths into prompts.
- Define a quality-evaluation workflow for generated results, including at least input coverage, extracted-question accuracy, answer usefulness, and grounding quality.
- Add a lightweight evaluation dataset or review checklist for manual spot checks on generated question and answer quality.
- Tighten normalization failure handling so malformed structured output can be retried or inspected more clearly.
- Add Markdown export for generated answers in addition to `answers.json`.
- Add Telegram dispatch for concise answers and note summaries.
- Reconstruct generated answers into a human-friendly Markdown archive format before local storage and Telegram reply.
- Store all normalized questions and generated answers in vector storage, not only question embeddings, so later retrieval and archival workflows can reuse both sides.
- Add bot-loop orchestration, retry tracking, and failure alerts.
- Add pipeline status persistence for discovery, extraction, normalization, indexing, generation, and dispatch steps.
- Add lightweight tests for storage and workflow glue code where external APIs are not required.

## Nice-to-have later items

- Keep the current web favorites collector as an optional enrichment path rather than the main ingestion route.
- Improve favorites discovery auth failure detection so expired sessions are reported clearly instead of appearing as an empty scan.
- Handle Xiaohongshu risk-control pages such as error code `300012` explicitly and surface them as actionable failures.
- Handle note-detail access failures such as error code `300031` and distinguish them from true extraction bugs.
- Add a reliable network/proxy strategy for Xiaohongshu access when the current environment is blocked by IP risk controls.
- Replace the fixed wait in favorites discovery with smarter waiting and optional scrolling for lazy-loaded content.
- Narrow the favorites note selector from the current broad default to a page-specific selector.
- Enforce designated-folder validation more explicitly instead of relying only on `XHS_FAVORITES_URL`.
- Expand note-id extraction if Xiaohongshu uses additional valid note URL patterns.
- Add fallback ingestion paths for notes that are visible in lists but blocked on the web detail page.
- Add pagination or incremental scrolling for large favorites folders.
- Add a local search or review interface on top of stored outputs.
- Add retrieval quality tuning such as reranking or better query construction if similarity quality is not good enough.
- Evaluate whether Xiaohongshu in-app private delivery is worth the extra automation risk.
