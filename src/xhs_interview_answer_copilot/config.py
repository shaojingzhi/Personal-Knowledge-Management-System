import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default
    if parsed_value <= 0:
        return default
    return parsed_value


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _read_optional_env(name: str) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    stripped_value = raw_value.strip()
    return stripped_value or None


@dataclass(frozen=True)
class Settings:
    xhs_browser_mode: str
    xhs_cdp_url: str | None
    xhs_base_url: str
    xhs_profile_dir: str
    xhs_favorites_url: str | None
    xhs_favorites_folder_name: str
    xhs_authenticated_selector: str | None
    xhs_note_link_selector: str
    xhs_login_timeout_seconds: int
    output_dir: str
    openai_api_key: str | None
    openai_base_url: str | None
    openai_proxy_url: str | None
    openai_timeout_seconds: int
    normalize_model: str
    embedding_model: str
    answer_model: str
    retrieval_top_k: int
    retrieval_min_score: float
    telegram_api_base: str
    telegram_proxy_url: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    sqlite_path: str


def load_settings() -> Settings:
    return Settings(
        xhs_browser_mode=os.getenv("XHS_BROWSER_MODE", "persistent"),
        xhs_cdp_url=_read_optional_env("XHS_CDP_URL"),
        xhs_base_url=os.getenv("XHS_BASE_URL", "https://www.xiaohongshu.com"),
        xhs_profile_dir=os.getenv("XHS_PROFILE_DIR", "./data/xhs-profile"),
        xhs_favorites_url=_read_optional_env("XHS_FAVORITES_URL"),
        xhs_favorites_folder_name=os.getenv(
            "XHS_FAVORITES_FOLDER_NAME", "interview-favorites"
        ),
        xhs_authenticated_selector=_read_optional_env("XHS_AUTHENTICATED_SELECTOR"),
        xhs_note_link_selector=os.getenv("XHS_NOTE_LINK_SELECTOR", "a"),
        xhs_login_timeout_seconds=_read_int_env("XHS_LOGIN_TIMEOUT_SECONDS", 300),
        output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
        openai_api_key=_read_optional_env("OPENAI_API_KEY"),
        openai_base_url=_read_optional_env("OPENAI_BASE_URL"),
        openai_proxy_url=_read_optional_env("OPENAI_PROXY_URL"),
        openai_timeout_seconds=_read_int_env("OPENAI_TIMEOUT_SECONDS", 60),
        normalize_model=os.getenv("NORMALIZE_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        answer_model=os.getenv("ANSWER_MODEL", "gpt-4.1-mini"),
        retrieval_top_k=_read_int_env("RETRIEVAL_TOP_K", 3),
        retrieval_min_score=_read_float_env("RETRIEVAL_MIN_SCORE", 0.25),
        telegram_api_base=os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org"),
        telegram_proxy_url=_read_optional_env("TELEGRAM_PROXY_URL"),
        telegram_bot_token=_read_optional_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_read_optional_env("TELEGRAM_CHAT_ID"),
        sqlite_path=os.getenv("SQLITE_PATH", "./data/app.db"),
    )


settings = load_settings()
