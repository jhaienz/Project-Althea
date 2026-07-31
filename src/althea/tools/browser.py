"""Persistent Playwright browser automation for web-based tasks."""

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

_PROFILE_DIR = Path.home() / ".local/share/althea/browser-profile"


class BrowserTool:
    """Manage one persistent Playwright browser context and its active page."""

    def __init__(
        self,
        page: Any | None = None,
        playwright_factory: Callable[[], Any] = async_playwright,
        profile_dir: Path = _PROFILE_DIR,
    ) -> None:
        self._page = page
        self._playwright_factory = playwright_factory
        self._profile_dir = profile_dir
        self._playwright: Any | None = None
        self._context: Any | None = None

    @property
    def page(self) -> Any:
        if self._page is None:
            raise RuntimeError("Browser has not started.")
        return self._page

    async def start(self) -> str:
        if self._page is None:
            self._profile_dir.mkdir(parents=True, exist_ok=True)
            playwright = await self._playwright_factory().start()
            context = await playwright.chromium.launch_persistent_context(
                self._profile_dir, headless=False
            )
            self._playwright = playwright
            self._context = context
            self._page = context.pages[0] if context.pages else await context.new_page()
        return "Browser ready."

    async def stop(self) -> str:
        if self._context is not None:
            await self._context.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._page = self._context = self._playwright = None
        return "Browser closed."

    async def navigate(self, url: str) -> str:
        target = url.strip()
        if not target:
            return "Please specify a URL to open."
        if not urlsplit(target).scheme:
            target = f"https://{target}"
        if urlsplit(target).scheme not in {"http", "https"}:
            return "Only HTTP and HTTPS URLs can be opened."
        await self.start()
        await self.page.goto(target)
        return f"Opened {target}."

    async def click(self, selector: str) -> str:
        await self.start()
        await self.page.locator(selector).click()
        return f"Clicked {selector}."

    async def fill(self, selector: str, text: str) -> str:
        await self.start()
        await self.page.locator(selector).fill(text)
        return f"Filled {selector}."

    async def read(self, selector: str = "body") -> str:
        await self.start()
        return (await self.page.locator(selector).inner_text()).strip()

    async def send_message(self, selector: str, message: str) -> str:
        await self.start()
        target = self.page.locator(selector)
        await target.fill(message)
        await target.press("Enter")
        return "Message sent."

    async def native_type(self, text: str) -> str:
        if os.getenv("XDG_SESSION_TYPE", "").casefold() != "wayland":
            return "ydotool fallback is only available on Wayland."
        executable = shutil.which("ydotool")
        if executable is None:
            return "ydotool is not installed."
        try:
            await asyncio.to_thread(
                subprocess.run, [executable, "type", "--", text], check=True
            )
        except (OSError, subprocess.CalledProcessError) as error:
            return f"ydotool could not type the text: {error}"
        return "Typed text with ydotool."


_browser = BrowserTool()


async def browser_start() -> str:
    """Start or reuse Althea's persistent browser session."""
    return await _browser.start()


async def browser_stop() -> str:
    """Close Althea's browser session cleanly."""
    return await _browser.stop()


async def browser_navigate(url: str) -> str:
    """Open an HTTP or HTTPS URL in Althea's persistent browser."""
    return await _browser.navigate(url)


async def browser_click(selector: str) -> str:
    """Click the web page element matching a Playwright selector."""
    return await _browser.click(selector)


async def browser_fill(selector: str, text: str) -> str:
    """Fill a web page field matching a Playwright selector."""
    return await _browser.fill(selector, text)


async def browser_read(selector: str = "body") -> str:
    """Read text from the web page element matching a Playwright selector."""
    return await _browser.read(selector)


async def browser_send_message(selector: str, message: str) -> str:
    """Fill a web message field and press Enter to send it."""
    return await _browser.send_message(selector, message)


async def native_type(text: str) -> str:
    """As a last resort, type into a native Wayland app after other Tools fail."""
    return await _browser.native_type(text)
