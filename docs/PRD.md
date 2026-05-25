# Knowledge Management Agent PRD v3

## Overview

Build a Telegram-first study assistant that takes text sent from mobile, automatically parses interview questions, generates answers, writes a readable Markdown archive locally, stores question-answer records for retrieval and later organization, and sends the generated answer text back to Telegram.

## Product Goal

Let a job-seeking developer send interview content to a bot and get a reusable local-and-mobile study artifact back with minimal manual steps.

## Success Criteria

- A user sends a text message to the Telegram Bot.
- The system ingests the message and stores a replayable raw bundle locally.
- The system normalizes the text into structured interview questions.
- The system generates usable answers for those questions.
- The system writes a readable Markdown answer file locally.
- The system stores question-answer records for later retrieval and archival use.
- The system sends a text answer summary back to Telegram.

## Core Scope

- Telegram Bot as the primary ingestion and delivery channel.
- Text-first `SourceBundle` ingestion from Telegram.
- Automatic one-shot processing from new Telegram text message to final outputs.
- LangGraph-based normalization and answer generation.
- Local artifact output including `raw.json`, `normalized.json`, `answers.json`, and `answer.md`.
- Vector storage for both questions and generated answers.
- Telegram outbound reply using generated answer text.

## Optional Scope

- OCR or multimodal handling for screenshot-heavy inputs.
- Optional web enrichment for reachable Xiaohongshu links.
- Message grouping or multi-message merge for split Telegram inputs.
- Additional source adapters such as Feishu.
- Bot-loop orchestration, retries, and detailed task-state tracking.

## Non-Goals

- Fully autonomous Xiaohongshu private messaging in phase one.
- Depending on Xiaohongshu web accessibility for the core text-first path.
- Building a full UI before the Telegram text path is stable.

## Users

- Primary user: a job-seeking developer collecting interview content for later review.

## Core Workflow

1. User sends interview text to a Telegram Bot.
2. The system stores the incoming message as a replayable `SourceBundle`.
3. The system normalizes the text into structured interview questions.
4. The system stores normalized question records so later bundles can retrieve them for grounding.
5. The system retrieves similar prior records for grounding.
6. The system generates concise and detailed answers.
7. The system writes local JSON artifacts and a readable Markdown answer file.
8. The system inserts question-answer records into vector storage for later retrieval.
9. The system sends the generated text answer summary back to Telegram.

## Technical Direction

- Python as the implementation language.
- Telegram Bot API as the primary inbound and outbound channel.
- `SourceBundle` as the shared source abstraction.
- LangGraph for normalization and answer-generation workflows.
- Pydantic for typed schemas.
- SQLite for lightweight local persistence and state.
- Local-first vector storage with fallback embeddings when provider embeddings are unavailable.
- Retrieval uses prior stored records when available; the first processed bundle may generate without useful history.
- Markdown artifacts as the default human-readable archive format.

## Risks

- Telegram and model-provider access may require a proxy.
- Text-only inputs can still be too sparse or low quality to produce trustworthy answers.
- Provider model compatibility can vary for structured output and embeddings.
- Quality can degrade if messages contain only link previews or incomplete question context.

## Ralph-Oriented Scope Split

### Core Stories

1. Bootstrap project and config loading.
2. Ingest Telegram text messages into `SourceBundle`.
3. Persist raw Telegram bundles locally.
4. Normalize text bundles into structured interview questions.
5. Generate detailed and concise answers from normalized questions.
6. Persist local JSON and Markdown artifacts.
7. Store question-answer records in vector storage.
8. Reply to Telegram with generated answer text.
9. Add one-shot automatic orchestration from ingestion to final outputs.

### Optional Stories

10. Add OCR or multimodal media extraction for screenshot-first bundles.
11. Add optional web-side enrichment for reachable Xiaohongshu links.
12. Add Telegram message grouping for split or related inputs.
13. Add another source adapter such as Feishu using `SourceBundle`.
14. Add continuous bot-loop orchestration, retries, and task tracking.
