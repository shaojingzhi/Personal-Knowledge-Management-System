# XHS Interview Answer Copilot

Personal automation for turning newly favorited Xiaohongshu interview posts into structured questions and AI-generated answers.

## Planned stack

- Python
- Playwright
- LangGraph
- Pydantic
- SQLite
- Telegram Bot API

## Current status

- PRD drafted in `docs/PRD.md`
- Ralph execution plan drafted in `scripts/ralph/prd.json`

## Package layout

- `src/xhs_interview_answer_copilot/collectors`
- `src/xhs_interview_answer_copilot/workflows`
- `src/xhs_interview_answer_copilot/storage`
- `src/xhs_interview_answer_copilot/dispatch`

## Configuration

Bootstrap-stage environment variables:

- `XHS_BASE_URL`
- `XHS_BROWSER_MODE`
- `XHS_CDP_URL`
- `XHS_PROFILE_DIR`
- `XHS_FAVORITES_URL`
- `XHS_FAVORITES_FOLDER_NAME`
- `XHS_AUTHENTICATED_SELECTOR`
- `XHS_NOTE_LINK_SELECTOR`
- `XHS_LOGIN_TIMEOUT_SECONDS`
- `OUTPUT_DIR`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_PROXY_URL`
- `OPENAI_TIMEOUT_SECONDS`
- `NORMALIZE_MODEL`
- `VISION_MODEL`
- `EMBEDDING_MODEL`
- `ANSWER_MODEL`
- `RETRIEVAL_MODE`
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_MIN_SCORE`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAM_API_BASE`
- `TELEGRAM_PROXY_URL`
- `SQLITE_PATH`

## Run

After installing the package in editable mode, run:

- `xhs-copilot status`
- `xhs-copilot login`
- `xhs-copilot healthcheck`
- `xhs-copilot discover`
- `xhs-copilot extract-note <note_id>`
- `xhs-copilot normalize-note <note_id>`
- `xhs-copilot index-note <note_id>`
- `xhs-copilot search-similar <query>`
- `xhs-copilot generate-answers <note_id>`
- `xhs-copilot generate-answers <note_id> --quick`
- `xhs-copilot backfill-full-answers <note_id>`
- `xhs-copilot store-answer-records <note_id>`
- `xhs-copilot reply-telegram <note_id>`
- `xhs-copilot process-telegram-once`
- `xhs-copilot process-telegram-daemon --interval-seconds 15`
- `xhs-copilot telegram-worker-start`
- `xhs-copilot telegram-worker-status`
- `xhs-copilot telegram-worker-stop`
- `xhs-copilot ingest-feishu-event <json_path>`
- `xhs-copilot ingest-telegram-once`
- or `python -m xhs_interview_answer_copilot.main status`

## Playwright setup

Install Python dependencies and browser binaries before using login commands:

- `pip install -e .`
- `playwright install chromium`

`XHS_AUTHENTICATED_SELECTOR` is optional but strongly recommended. If you set it to a selector that is only visible after login, `xhs-copilot login` can wait for it during bootstrap and `xhs-copilot healthcheck` can verify the saved session more reliably. Without it, the login command falls back to manual confirmation and health checks return conservative results when they cannot prove the session is still authenticated.

For a lower-risk setup, set `XHS_BROWSER_MODE=cdp`, launch your own Chrome with `--remote-debugging-port=9222`, keep your normal Xiaohongshu session logged in there, and set `XHS_CDP_URL=http://127.0.0.1:9222`.

`XHS_FAVORITES_URL` should point to the exact favorites page you want to scan. `XHS_NOTE_LINK_SELECTOR` defaults to `a` for a minimal MVP and can be narrowed later if the page contains too many unrelated links.

`xhs-copilot extract-note <note_id>` looks up the note URL from the local discovery database and writes a per-note `raw.json` under `OUTPUT_DIR`.

`xhs-copilot normalize-note <note_id>` is the first LangGraph-based step. It loads `raw.json` from either a web note bundle or a Telegram bundle, calls an LLM through LangChain structured output, and writes `normalized.json` under `OUTPUT_DIR`.

`xhs-copilot index-note <note_id>` embeds the normalized questions and stores them in a local SQLite-backed vector table. `xhs-copilot search-similar <query>` runs retrieval over those indexed questions using the current retrieval mode.

`RETRIEVAL_MODE` supports `vector`, `bm25`, and `hybrid`. `vector` uses embeddings, `bm25` uses local text scoring over the indexed question corpus, and `hybrid` merges both rankings. The active default mode can also be changed from Telegram by sending a message that is exactly `[vector]`, `[bm25]`, or `[hybrid]`.

`xhs-copilot generate-answers <note_id>` is the first end-to-end RAG answer step. It loads `normalized.json`, retrieves similar indexed questions, and writes `answers.json` under `OUTPUT_DIR`.

`xhs-copilot store-answer-records <note_id>` stores generated question-answer records in local vector storage so later retrieval and archival flows can reuse both questions and answers.

`xhs-copilot reply-telegram <note_id>` sends a concise text reply built from the generated answers back to the configured Telegram chat.

`xhs-copilot process-telegram-once` is the LangGraph-based one-shot orchestration command. It ingests Telegram once, then automatically runs normalization, question indexing, answer generation, Markdown archival, question-answer vector storage, and Telegram reply for each new bundle.

Telegram orchestration now sends progress messages while it OCRs, normalizes, indexes, generates, stores, and replies, so large image notes do not look stuck. It sends a quick local answer first so image notes can be acknowledged quickly. A non-blocking `backfill-full-answers` process is started after the Telegram reply to regenerate full LLM answers, refresh stored answer records when possible, and send the completed `answer.md` file back to Telegram as a document.

`xhs-copilot process-telegram-daemon` runs the same Telegram pipeline in a loop so the project can behave like a resident local worker. Use `--interval-seconds` to control the normal polling cadence, `--failure-backoff-seconds` to slow down after failures such as provider rate limits, and `--max-loops` for local testing.

For a practical background worker, run `xhs-copilot telegram-worker-start`. It starts `process-telegram-daemon` inside a detached tmux session named `xhs-copilot-telegram`, appends logs to `telegram-daemon.log` next to `SQLITE_PATH` (default `data/telegram-daemon.log`), and avoids launching a duplicate worker if the session already exists. Use `xhs-copilot telegram-worker-status` to check whether it is running, `tmux attach -t xhs-copilot-telegram` to inspect it interactively, and `xhs-copilot telegram-worker-stop` to stop it.

`xhs-copilot ingest-feishu-event <json_path>` ingests a Feishu event payload file, maps supported text-message events into the shared `SourceBundle` schema, and stores the resulting raw bundle locally.

`xhs-copilot ingest-telegram-once` fetches Telegram updates once, downloads any attached photo or document from allowed chats, and writes a raw bundle under `OUTPUT_DIR/telegram_<update_id>/raw.json`.

Raw ingestion now trends toward a shared `SourceBundle` shape with explicit `canonical_url`, `text_blocks`, `asset_paths`, and `image_urls`, so Telegram is the first message-driven source and later integrations such as Feishu can target the same normalization entry point.

If `api.telegram.org` is blocked in your network, set `TELEGRAM_PROXY_URL` to an HTTP or HTTPS proxy address such as `http://127.0.0.1:7890`.

If model requests need the same proxy, set `OPENAI_PROXY_URL` such as `http://127.0.0.1:7890`. You can also adjust `OPENAI_TIMEOUT_SECONDS` for slower model responses.

For quick local testing when your endpoint does not provide embeddings, set `EMBEDDING_MODEL=local-hash-v1` to use a deterministic local hash embedding fallback.

If your endpoint supports multimodal input, set `VISION_MODEL` to a compatible model name so screenshot assets can be converted into text during normalization.
