from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final, Iterator

from xhs_interview_answer_copilot.config import Settings

LOGIN_INDICATOR_TEXTS: Final[tuple[str, ...]] = (
    "登录",
    "扫码登录",
    "手机号登录",
    "立即登录",
)
RISK_CONTROL_TEXTS: Final[tuple[str, ...]] = (
    "安全限制",
    "IP存在风险",
    "error_code=300012",
)


@dataclass(frozen=True)
class AuthCheckResult:
    is_authenticated: bool
    reason: str
    current_url: str


class BrowserSessionManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def bootstrap_login(self) -> AuthCheckResult:
        try:
            with self.open_context() as context:
                page = None
                try:
                    page = context.new_page()
                    page.goto(self._settings.xhs_base_url, wait_until="domcontentloaded")
                    print(self._bootstrap_message())
                    print("Complete QR login in the browser window if prompted.")
                    if self._settings.xhs_authenticated_selector:
                        print(
                            "Waiting for authenticated selector up to "
                            f"{self._settings.xhs_login_timeout_seconds} seconds."
                        )
                    else:
                        print("Press Enter after login finishes to run a session check.")
                    result = self._wait_for_login(page=page)
                    return result
                finally:
                    if page is not None:
                        page.close()
        except Exception as exc:
            return AuthCheckResult(
                is_authenticated=False,
                reason=f"Failed to open browser session: {exc}",
                current_url=self._settings.xhs_base_url,
            )

    def check_authentication(self) -> AuthCheckResult:
        try:
            with self.open_context() as context:
                page = None
                try:
                    page = context.new_page()
                    page.goto(self._settings.xhs_base_url, wait_until="domcontentloaded")
                    return self._inspect_page_auth(page=page)
                finally:
                    if page is not None:
                        page.close()
        except Exception as exc:
            return AuthCheckResult(
                is_authenticated=False,
                reason=f"Failed to inspect browser session: {exc}",
                current_url=self._settings.xhs_base_url,
            )

    def inspect_page_auth(self, page: Any) -> AuthCheckResult:
        return self._inspect_page_auth(page=page)

    @contextmanager
    def open_context(self) -> Iterator[Any]:
        sync_api = import_module("playwright.sync_api")
        sync_playwright = sync_api.sync_playwright
        Error = sync_api.Error
        with sync_playwright() as playwright:
            chromium = getattr(playwright, "chromium")
            if self._settings.xhs_browser_mode == "cdp":
                if self._settings.xhs_cdp_url is None:
                    raise Error("XHS_CDP_URL is required when XHS_BROWSER_MODE=cdp")
                browser = chromium.connect_over_cdp(self._settings.xhs_cdp_url)
                context = browser.contexts[0] if browser.contexts else browser.new_context()
                yield context
                return

            profile_dir = Path(self._settings.xhs_profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            context = chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=False,
            )
            try:
                yield context
            finally:
                context.close()

    def _bootstrap_message(self) -> str:
        if self._settings.xhs_browser_mode == "cdp":
            return "Connected to existing Chrome via CDP."
        return f"Opened Xiaohongshu in persistent profile: {self._settings.xhs_profile_dir}"

    def _wait_for_login(self, page: Any) -> AuthCheckResult:
        sync_api = import_module("playwright.sync_api")
        TimeoutError = sync_api.TimeoutError
        if self._settings.xhs_authenticated_selector:
            try:
                page.wait_for_selector(
                    self._settings.xhs_authenticated_selector,
                    timeout=self._settings.xhs_login_timeout_seconds * 1000,
                )
                return self._inspect_page_auth(page=page)
            except TimeoutError:
                return self._inspect_page_auth(page=page)

        try:
            input()
        except EOFError:
            return AuthCheckResult(
                is_authenticated=False,
                reason=(
                    "No authenticated selector configured and no interactive input was "
                    "available to confirm login completion."
                ),
                current_url=page.url,
            )
        return self._inspect_page_auth(page=page)

    def _inspect_page_auth(self, page: Any) -> AuthCheckResult:
        sync_api = import_module("playwright.sync_api")
        Error = sync_api.Error
        TimeoutError = sync_api.TimeoutError
        selector = self._settings.xhs_authenticated_selector
        if selector:
            locator = page.locator(self._settings.xhs_authenticated_selector)
            try:
                if locator.first.is_visible(timeout=3_000):
                    return AuthCheckResult(
                        is_authenticated=True,
                        reason="Authenticated selector is visible.",
                        current_url=page.url,
                    )
            except (TimeoutError, Error):
                pass

        try:
            body_text = page.locator("body").inner_text(timeout=5_000)
        except (TimeoutError, Error) as exc:
            return AuthCheckResult(
                is_authenticated=False,
                reason=f"Could not inspect page content reliably: {exc}",
                current_url=page.url,
            )
        lowered_text = body_text.lower()
        if any(indicator in body_text for indicator in RISK_CONTROL_TEXTS):
            return AuthCheckResult(
                is_authenticated=False,
                reason="Detected Xiaohongshu risk-control page.",
                current_url=page.url,
            )
        if any(indicator in body_text for indicator in LOGIN_INDICATOR_TEXTS):
            return AuthCheckResult(
                is_authenticated=False,
                reason="Detected login prompts in page content.",
                current_url=page.url,
            )
        if "login" in page.url.lower() or "signin" in page.url.lower() or "登录" in lowered_text:
            return AuthCheckResult(
                is_authenticated=False,
                reason="Detected login-related url or text.",
                current_url=page.url,
            )
        if self._settings.xhs_authenticated_selector is None:
            return AuthCheckResult(
                is_authenticated=False,
                reason=(
                    "No login prompts were detected, but no authenticated selector is "
                    "configured, so the session cannot be verified confidently."
                ),
                current_url=page.url,
            )
        return AuthCheckResult(
            is_authenticated=False,
            reason=(
                "Authenticated selector is configured but was not found, so the session "
                "cannot be verified as logged in."
            ),
            current_url=page.url,
        )
