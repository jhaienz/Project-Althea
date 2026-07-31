"""Tests for the Gmail Tool (issue #10)."""

import base64
from email import policy
from email import message_from_bytes
from unittest.mock import MagicMock, patch

from althea.tools.gmail import GmailTool


def _service() -> tuple[MagicMock, MagicMock]:
    service = MagicMock()
    messages = service.users.return_value.messages.return_value
    return service, messages


def test_check_email_lists_and_summarizes_important_messages() -> None:
    service, messages = _service()
    messages.list.return_value.execute.return_value = {"messages": [{"id": "m1"}]}
    messages.get.return_value.execute.return_value = {
        "id": "m1",
        "snippet": "The build passed successfully.",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Build report"},
                {"name": "From", "value": "CI <ci@example.com>"},
            ]
        },
    }

    result = GmailTool(service=service).check()

    messages.list.assert_called_once_with(
        userId="me", q="is:important", maxResults=5
    )
    messages.get.assert_called_once_with(
        userId="me",
        id="m1",
        format="metadata",
        metadataHeaders=["Subject", "From"],
    )
    assert result == (
        "Subject: Build report; From: CI <ci@example.com>; "
        "Summary: The build passed successfully."
    )


def test_read_email_returns_plain_text_content() -> None:
    service, messages = _service()
    body = base64.urlsafe_b64encode(b"A").decode().rstrip("=")
    messages.get.return_value.execute.return_value = {
        "payload": {"mimeType": "text/plain", "body": {"data": body}}
    }

    result = GmailTool(service=service).read("m1")

    messages.get.assert_called_once_with(userId="me", id="m1", format="full")
    assert result == "A"


def test_search_email_passes_search_text_to_gmail() -> None:
    service, messages = _service()
    messages.list.return_value.execute.return_value = {"messages": []}

    result = GmailTool(service=service).search("from:ci@example.com", limit=3)

    messages.list.assert_called_once_with(
        userId="me", q="from:ci@example.com", maxResults=3
    )
    assert result == "No matching emails found."


def test_send_email_composes_rfc_message() -> None:
    service, messages = _service()
    messages.send.return_value.execute.return_value = {"id": "sent-1"}

    result = GmailTool(service=service).send(
        "mom@example.com", "Hello", "Just checking in."
    )

    messages.send.assert_called_once()
    call = messages.send.call_args.kwargs
    assert call["userId"] == "me"
    email = message_from_bytes(
        base64.urlsafe_b64decode(call["body"]["raw"]), policy=policy.default
    )
    assert email["To"] == "mom@example.com"
    assert email["Subject"] == "Hello"
    assert email.get_content().strip() == "Just checking in."
    assert result == "Sent email to mom@example.com."


def test_first_connection_authorizes_and_stores_credentials() -> None:
    store = MagicMock()
    store.load.return_value = None
    credentials = MagicMock(valid=True)
    credentials.to_json.return_value = '{"token": "new"}'
    flow = MagicMock()
    flow.run_local_server.return_value = credentials
    service = MagicMock()

    with (
        patch(
            "althea.tools.gmail.InstalledAppFlow.from_client_secrets_file",
            return_value=flow,
        ) as from_file,
        patch("althea.tools.gmail.build", return_value=service) as build,
    ):
        result = GmailTool(token_store=store).connect()

    from_file.assert_called_once()
    flow.run_local_server.assert_called_once_with(port=0)
    store.save.assert_called_once_with('{"token": "new"}')
    build.assert_called_once_with("gmail", "v1", credentials=credentials)
    assert result == "Gmail connected."


def test_expired_credentials_refresh_without_authorizing_again() -> None:
    store = MagicMock()
    store.load.return_value = '{"token": "old"}'
    credentials = MagicMock(valid=False, expired=True, refresh_token="refresh")
    credentials.to_json.return_value = '{"token": "fresh"}'

    with (
        patch(
            "althea.tools.gmail.Credentials.from_authorized_user_info",
            return_value=credentials,
        ),
        patch("althea.tools.gmail.Request") as request,
        patch("althea.tools.gmail.InstalledAppFlow") as flow,
        patch("althea.tools.gmail.build"),
    ):
        GmailTool(token_store=store).connect()

    credentials.refresh.assert_called_once_with(request.return_value)
    store.save.assert_called_once_with('{"token": "fresh"}')
    flow.from_client_secrets_file.assert_not_called()
