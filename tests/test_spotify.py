"""Tests for the Spotify Tool (issue #9)."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from althea.tools.spotify import SpotifyTool


@pytest.fixture()
def client() -> MagicMock:
    spotify = MagicMock()
    spotify.search.return_value = {
        "tracks": {
            "items": [
                {
                    "name": "Lofi Study",
                    "uri": "spotify:track:1",
                    "artists": [{"name": "Chill Artist"}],
                }
            ]
        }
    }
    return spotify


def test_play_searches_and_starts_the_first_match(client: MagicMock) -> None:
    result = SpotifyTool(client).play("lo-fi")

    client.search.assert_called_once_with(q="lo-fi", type="track", limit=1)
    client.start_playback.assert_called_once_with(uris=["spotify:track:1"])
    assert result == "Playing Lofi Study by Chill Artist."


@pytest.mark.parametrize(
    ("action", "api_method", "message"),
    [
        ("pause", "pause_playback", "Paused Spotify."),
        ("skip", "next_track", "Skipped to the next song."),
        ("previous", "previous_track", "Playing the previous song."),
    ],
)
def test_playback_controls_call_spotify(
    client: MagicMock, action: str, api_method: str, message: str
) -> None:
    result = getattr(SpotifyTool(client), action)()

    getattr(client, api_method).assert_called_once_with()
    assert result == message


def test_queue_searches_and_adds_the_first_match(client: MagicMock) -> None:
    result = SpotifyTool(client).queue("lo-fi")

    client.search.assert_called_once_with(q="lo-fi", type="track", limit=1)
    client.add_to_queue.assert_called_once_with("spotify:track:1")
    assert result == "Queued Lofi Study by Chill Artist."


def test_search_returns_track_names_and_artists(client: MagicMock) -> None:
    result = SpotifyTool(client).search("artist:Chill Artist")

    client.search.assert_called_once_with(
        q="artist:Chill Artist", type="track", limit=5
    )
    assert result == "Lofi Study by Chill Artist"


def test_current_track_reports_name_and_artist(client: MagicMock) -> None:
    client.current_user_playing_track.return_value = {
        "item": {"name": "Lofi Study", "artists": [{"name": "Chill Artist"}]}
    }

    result = SpotifyTool(client).current_track()

    client.current_user_playing_track.assert_called_once_with()
    assert result == "Lofi Study by Chill Artist is playing."


def test_current_track_handles_idle_spotify(client: MagicMock) -> None:
    client.current_user_playing_track.return_value = None

    assert SpotifyTool(client).current_track() == "Nothing is playing on Spotify."


def test_first_connection_uses_refreshable_file_backed_oauth(tmp_path: Path) -> None:
    token_path = tmp_path / "spotify-token.json"
    client = MagicMock()

    with (
        patch("althea.tools.spotify._TOKEN_PATH", token_path),
        patch("althea.tools.spotify.CacheFileHandler") as cache_handler,
        patch("althea.tools.spotify.SpotifyOAuth") as oauth,
        patch("althea.tools.spotify.spotipy.Spotify", return_value=client) as spotify,
    ):
        result = SpotifyTool().client

    cache_handler.assert_called_once_with(cache_path=str(token_path))
    oauth.assert_called_once_with(
        scope=(
            "user-modify-playback-state user-read-playback-state "
            "user-read-currently-playing"
        ),
        cache_handler=cache_handler.return_value,
    )
    spotify.assert_called_once_with(auth_manager=oauth.return_value)
    assert result is client
