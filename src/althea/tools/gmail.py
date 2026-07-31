"""Gmail reading and sending with OAuth tokens stored in GNOME Keyring."""

import base64
import json
import os
from email.message import EmailMessage
from typing import Any

import secretstorage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
)
_KEYRING_ATTRIBUTES = {"application": "althea", "account": "gmail"}


class GnomeKeyringTokenStore:
    """Store the Gmail OAuth token in the user's default GNOME Keyring."""

    def load(self) -> str | None:
        connection = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(connection)
        if collection.is_locked():
            collection.unlock()
        item = next(collection.search_items(_KEYRING_ATTRIBUTES), None)
        return item.get_secret().decode() if item else None

    def save(self, token: str) -> None:
        connection = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(connection)
        if collection.is_locked():
            collection.unlock()
        collection.create_item(
            "Althea Gmail OAuth token",
            _KEYRING_ATTRIBUTES,
            token.encode(),
            replace=True,
        )


class GmailTool:
    """Persistent Gmail API client used by the Agent's email Tool."""

    def __init__(
        self,
        service: Any | None = None,
        token_store: GnomeKeyringTokenStore | Any | None = None,
    ) -> None:
        self._service = service
        self._token_store = token_store

    @property
    def service(self) -> Any:
        if self._service is None:
            store = self._token_store or GnomeKeyringTokenStore()
            token = store.load()
            credentials = (
                Credentials.from_authorized_user_info(json.loads(token), _SCOPES)
                if token
                else None
            )
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                else:
                    credentials_path = os.getenv(
                        "ALTHEA_GMAIL_CREDENTIALS_PATH", "credentials.json"
                    )
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, _SCOPES
                    )
                    credentials = flow.run_local_server(port=0)
                store.save(credentials.to_json())
            self._service = build("gmail", "v1", credentials=credentials)
        return self._service

    def connect(self) -> str:
        self.service
        return "Gmail connected."

    @staticmethod
    def _headers(message: dict[str, Any]) -> dict[str, str]:
        return {
            header["name"].casefold(): header["value"]
            for header in message.get("payload", {}).get("headers", [])
        }

    def _summaries(self, search_text: str, limit: int) -> list[str]:
        messages = self.service.users().messages()
        result = messages.list(
            userId="me", q=search_text, maxResults=limit
        ).execute()
        summaries = []
        for match in result.get("messages", []):
            message = messages.get(
                userId="me",
                id=match["id"],
                format="metadata",
                metadataHeaders=["Subject", "From"],
            ).execute()
            headers = self._headers(message)
            summaries.append(
                f"Message ID: {message['id']}; "
                f"Subject: {headers.get('subject', '(no subject)')}; "
                f"From: {headers.get('from', '(unknown sender)')}; "
                f"Summary: {message.get('snippet', '(no preview)')}"
            )
        return summaries

    def check(self, limit: int = 5) -> str:
        summaries = self._summaries("is:important", limit)
        return "; ".join(summaries) if summaries else "No important emails found."

    def recent(self, limit: int = 5) -> str:
        summaries = self._summaries("", limit)
        return "; ".join(summaries) if summaries else "No recent emails found."

    @classmethod
    def _plain_text(cls, payload: dict[str, Any]) -> str:
        if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get(
            "data"
        ):
            data = payload["body"]["data"]
            data += "=" * (-len(data) % 4)
            return base64.urlsafe_b64decode(data).decode(errors="replace")
        for part in payload.get("parts", []):
            if text := cls._plain_text(part):
                return text
        return ""

    def read(self, message_id: str) -> str:
        message = (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return self._plain_text(message.get("payload", {})) or message.get(
            "snippet", "This email has no readable text content."
        )

    def search(self, search_text: str, limit: int = 5) -> str:
        summaries = self._summaries(search_text, limit)
        return "; ".join(summaries) if summaries else "No matching emails found."

    def send(self, recipient: str, subject: str, body: str) -> str:
        if not recipient.strip() or not subject.strip() or not body.strip():
            return "Recipient, subject, and message are required."
        message = EmailMessage()
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        self.service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        return f"Sent email to {recipient}."


_gmail = GmailTool()


def connect_gmail() -> str:
    """Authorize Gmail on first use or refresh its saved OAuth token."""
    return _gmail.connect()


def check_email(limit: int = 5) -> str:
    """Summarize recent important emails by subject, sender, and brief content."""
    return _gmail.check(limit)


def recent_email(limit: int = 5) -> str:
    """Summarize recent emails by message ID, subject, sender, and brief content."""
    return _gmail.recent(limit)


def read_email(message_id: str) -> str:
    """Read the plain-text content of a Gmail message by its message ID."""
    return _gmail.read(message_id)


def search_email(search_text: str, limit: int = 5) -> str:
    """Search Gmail and summarize matching emails."""
    return _gmail.search(search_text, limit)


def send_email(recipient: str, subject: str, body: str) -> str:
    """Compose and send an email through Gmail."""
    return _gmail.send(recipient, subject, body)
