"""Spotify playback control through the Spotify Web API."""

from pathlib import Path
from typing import Any

import spotipy
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

_SCOPES = (
    "user-modify-playback-state user-read-playback-state "
    "user-read-currently-playing"
)
_TOKEN_PATH = Path.home() / ".local/share/althea/spotify-token.json"


class SpotifyTool:
    """Persistent Spotipy client used by the Agent's Spotify actions."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            auth = SpotifyOAuth(
                scope=_SCOPES,
                cache_handler=CacheFileHandler(cache_path=str(_TOKEN_PATH)),
            )
            self._client = spotipy.Spotify(auth_manager=auth)
        return self._client

    def _tracks(self, search_text: str, limit: int) -> list[dict[str, Any]]:
        result = self.client.search(q=search_text, type="track", limit=limit)
        return result.get("tracks", {}).get("items", [])

    @staticmethod
    def _describe(track: dict[str, Any]) -> str:
        artists = ", ".join(artist["name"] for artist in track.get("artists", []))
        return f"{track['name']} by {artists}" if artists else track["name"]

    def play(self, search_text: str) -> str:
        tracks = self._tracks(search_text, 1)
        if not tracks:
            return f"I couldn't find {search_text} on Spotify."
        track = tracks[0]
        self.client.start_playback(uris=[track["uri"]])
        return f"Playing {self._describe(track)}."

    def pause(self) -> str:
        self.client.pause_playback()
        return "Paused Spotify."

    def skip(self) -> str:
        self.client.next_track()
        return "Skipped to the next song."

    def previous(self) -> str:
        self.client.previous_track()
        return "Playing the previous song."

    def queue(self, search_text: str) -> str:
        tracks = self._tracks(search_text, 1)
        if not tracks:
            return f"I couldn't find {search_text} on Spotify."
        track = tracks[0]
        self.client.add_to_queue(track["uri"])
        return f"Queued {self._describe(track)}."

    def search(self, search_text: str) -> str:
        tracks = self._tracks(search_text, 5)
        if not tracks:
            return f"I couldn't find {search_text} on Spotify."
        return "; ".join(self._describe(track) for track in tracks)

    def current_track(self) -> str:
        playback = self.client.current_user_playing_track()
        if not playback or not playback.get("item"):
            return "Nothing is playing on Spotify."
        return f"{self._describe(playback['item'])} is playing."


_spotify = SpotifyTool()


def play_spotify(search_text: str) -> str:
    """Play the first Spotify track matching a song, artist, genre, or mood."""
    return _spotify.play(search_text)


def pause_spotify() -> str:
    """Pause Spotify playback."""
    return _spotify.pause()


def skip_spotify() -> str:
    """Skip to the next Spotify track."""
    return _spotify.skip()


def previous_spotify() -> str:
    """Return to the previous Spotify track."""
    return _spotify.previous()


def queue_spotify(search_text: str) -> str:
    """Add the first matching song, artist, genre, or mood to Spotify's queue."""
    return _spotify.queue(search_text)


def search_spotify(search_text: str) -> str:
    """Search Spotify for songs by title, artist, genre, or mood."""
    return _spotify.search(search_text)


def whats_playing() -> str:
    """Report the song and artist currently playing on Spotify."""
    return _spotify.current_track()
