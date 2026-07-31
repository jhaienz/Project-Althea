"""Tests for the app launcher Tool (issue #8)."""

from unittest.mock import patch

from althea.tools.app_launcher import launch_app


def test_installed_app_is_launched() -> None:
    with (
        patch(
            "althea.tools.app_launcher.shutil.which", return_value="/usr/bin/firefox"
        ) as which,
        patch("althea.tools.app_launcher.subprocess.Popen") as popen,
    ):
        result = launch_app("Firefox")

    which.assert_called_once_with("firefox")
    popen.assert_called_once_with(["/usr/bin/firefox"], start_new_session=True)
    assert result == "Opened Firefox"


def test_terminal_uses_default_terminal_emulator() -> None:
    with (
        patch(
            "althea.tools.app_launcher.shutil.which",
            side_effect=[None, "/usr/bin/gnome-terminal"],
        ) as which,
        patch("althea.tools.app_launcher.subprocess.Popen") as popen,
    ):
        result = launch_app("terminal")

    assert [call.args[0] for call in which.call_args_list] == [
        "x-terminal-emulator",
        "gnome-terminal",
    ]
    popen.assert_called_once_with(
        ["/usr/bin/gnome-terminal"], start_new_session=True
    )
    assert result == "Opened terminal"


def test_browser_uses_system_default() -> None:
    with (
        patch(
            "althea.tools.app_launcher.shutil.which", return_value="/usr/bin/xdg-open"
        ) as which,
        patch("althea.tools.app_launcher.subprocess.Popen") as popen,
    ):
        result = launch_app("browser")

    which.assert_called_once_with("xdg-open")
    popen.assert_called_once_with(
        ["/usr/bin/xdg-open", "about:blank"], start_new_session=True
    )
    assert result == "Opened browser"


def test_unknown_app_returns_helpful_error() -> None:
    with (
        patch("althea.tools.app_launcher.shutil.which", return_value=None),
        patch("althea.tools.app_launcher.subprocess.Popen") as popen,
    ):
        result = launch_app("Imaginary App")

    popen.assert_not_called()
    assert result == "Could not find Imaginary App. Make sure it is installed."


def test_launch_failure_returns_helpful_error() -> None:
    with (
        patch("althea.tools.app_launcher.shutil.which", return_value="/usr/bin/discord"),
        patch(
            "althea.tools.app_launcher.subprocess.Popen",
            side_effect=OSError("permission denied"),
        ),
    ):
        result = launch_app("Discord")

    assert result == "Could not open Discord: permission denied"


def test_empty_app_name_returns_helpful_error() -> None:
    with patch("althea.tools.app_launcher.subprocess.Popen") as popen:
        result = launch_app("  ")

    popen.assert_not_called()
    assert result == "Please specify an application to open."
