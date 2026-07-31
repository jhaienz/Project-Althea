"""Trivial tests to validate the test infrastructure."""

import pytest


def test_infrastructure_is_working() -> None:
    """Verify pytest can discover and run tests."""
    assert True


@pytest.mark.asyncio
async def test_async_infrastructure_is_working() -> None:
    """Verify pytest-asyncio can run async tests (needed for async Tool tests)."""
    result = await _async_noop()
    assert result == "ok"


async def _async_noop() -> str:
    return "ok"
