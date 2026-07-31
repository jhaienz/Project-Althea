"""Tests for the browser automation Tool (issue #11)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from althea.tools.browser import BrowserTool


def _page() -> tuple[MagicMock, MagicMock]:
    page = MagicMock()
    locator = MagicMock()
    page.goto = AsyncMock()
    page.locator.return_value = locator
    locator.click = AsyncMock()
    locator.fill = AsyncMock()
    locator.press = AsyncMock()
    locator.inner_text = AsyncMock()
    return page, locator


@pytest.mark.asyncio
async def test_navigate_adds_https_and_opens_the_url() -> None:
    page, _ = _page()

    result = await BrowserTool(page=page).navigate("github.com")

    page.goto.assert_called_once_with("https://github.com")
    assert result == "Opened https://github.com."


@pytest.mark.asyncio
async def test_navigate_rejects_an_empty_url() -> None:
    page, _ = _page()

    result = await BrowserTool(page=page).navigate("  ")

    page.goto.assert_not_awaited()
    assert result == "Please specify a URL to open."


@pytest.mark.asyncio
async def test_click_uses_a_page_locator() -> None:
    page, locator = _page()

    result = await BrowserTool(page=page).click("text=Sign in")

    page.locator.assert_called_once_with("text=Sign in")
    locator.click.assert_awaited_once_with()
    assert result == "Clicked text=Sign in."


@pytest.mark.asyncio
async def test_fill_types_into_a_page_element() -> None:
    page, locator = _page()

    result = await BrowserTool(page=page).fill("#email", "jai@example.com")

    page.locator.assert_called_once_with("#email")
    locator.fill.assert_awaited_once_with("jai@example.com")
    assert result == "Filled #email."


@pytest.mark.asyncio
async def test_read_returns_page_text() -> None:
    page, locator = _page()
    locator.inner_text.return_value = "  Page content  "

    result = await BrowserTool(page=page).read("main")

    page.locator.assert_called_once_with("main")
    assert result == "Page content"


@pytest.mark.asyncio
async def test_send_message_fills_and_submits() -> None:
    page, locator = _page()

    result = await BrowserTool(page=page).send_message(
        "[contenteditable]", "Hello Mom"
    )

    locator.fill.assert_awaited_once_with("Hello Mom")
    locator.press.assert_awaited_once_with("Enter")
    assert result == "Message sent."


@pytest.mark.asyncio
async def test_start_reuses_one_persistent_context(tmp_path: Path) -> None:
    playwright = MagicMock()
    context = MagicMock()
    page = MagicMock()
    context.pages = [page]
    playwright.chromium.launch_persistent_context = AsyncMock(return_value=context)
    manager = MagicMock()
    manager.start = AsyncMock(return_value=playwright)
    tool = BrowserTool(playwright_factory=lambda: manager, profile_dir=tmp_path)

    await tool.start()
    await tool.start()

    manager.start.assert_awaited_once_with()
    playwright.chromium.launch_persistent_context.assert_awaited_once_with(
        tmp_path, headless=False
    )
    assert tool.page is page


@pytest.mark.asyncio
async def test_stop_closes_context_and_playwright(tmp_path: Path) -> None:
    playwright = MagicMock()
    context = MagicMock(pages=[MagicMock()])
    context.close = AsyncMock()
    playwright.chromium.launch_persistent_context = AsyncMock(return_value=context)
    playwright.stop = AsyncMock()
    manager = MagicMock()
    manager.start = AsyncMock(return_value=playwright)
    tool = BrowserTool(playwright_factory=lambda: manager, profile_dir=tmp_path)
    await tool.start()

    await tool.stop()

    context.close.assert_awaited_once_with()
    playwright.stop.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_native_type_uses_ydotool_as_an_explicit_fallback() -> None:
    with (
        patch("althea.tools.browser.os.getenv", return_value="wayland"),
        patch("althea.tools.browser.shutil.which", return_value="/usr/bin/ydotool"),
        patch("althea.tools.browser.subprocess.run") as run,
    ):
        result = await BrowserTool().native_type("Hello Mom")

    run.assert_called_once_with(
        ["/usr/bin/ydotool", "type", "--", "Hello Mom"], check=True
    )
    assert result == "Typed text with ydotool."


@pytest.mark.asyncio
async def test_native_type_is_rejected_outside_wayland() -> None:
    with (
        patch("althea.tools.browser.os.getenv", return_value="x11"),
        patch("althea.tools.browser.subprocess.run") as run,
    ):
        result = await BrowserTool().native_type("Hello Mom")

    run.assert_not_called()
    assert result == "ydotool fallback is only available on Wayland."
