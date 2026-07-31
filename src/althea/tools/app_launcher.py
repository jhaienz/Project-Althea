"""Launch desktop applications without blocking Althea."""

import shutil
import subprocess

_TERMINAL_EXECUTABLES = (
    "x-terminal-emulator",
    "gnome-terminal",
    "konsole",
    "xfce4-terminal",
)
_DEFAULT_APP_COMMANDS = {"browser": ("xdg-open", "about:blank")}


def launch_app(app_name: str) -> str:
    """Launch an installed desktop application by name.

    Args:
        app_name: Common application name, such as Firefox or terminal.

    Returns:
        A short confirmation or a helpful error for Althea to speak.
    """
    display_name = app_name.strip()
    if not display_name:
        return "Please specify an application to open."
    name = display_name.lower()
    command = _DEFAULT_APP_COMMANDS.get(name, (name,))
    candidates = _TERMINAL_EXECUTABLES if name == "terminal" else command[:1]
    executable = next(
        (path for candidate in candidates if (path := shutil.which(candidate))),
        None,
    )
    if executable is None:
        return f"Could not find {display_name}. Make sure it is installed."
    try:
        subprocess.Popen([executable, *command[1:]], start_new_session=True)
    except OSError as error:
        return f"Could not open {display_name}: {error}"
    return f"Opened {display_name}"
