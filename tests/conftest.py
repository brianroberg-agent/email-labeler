"""Shared fixtures and sample Gmail data for tests."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def no_real_ntfy_credentials(monkeypatch):
    """No test may reach the real ntfy topic (review of #81).

    ``daemon.py`` calls ``load_dotenv()`` at import, so a populated ``.env`` in
    the checkout — never mind an exported shell variable — is enough for
    ``HaltNotifier.from_env()`` to return an *enabled* notifier pointed at the
    operator's real topic, and any test that then sends would POST to it.
    Clearing both variables for every test makes the notifier disabled unless
    the test sets them itself, in which case it is also responsible for
    patching the HTTP client.
    """
    monkeypatch.delenv("NTFY_URL", raising=False)
    monkeypatch.delenv("NTFY_TOKEN", raising=False)


@pytest.fixture
def mock_proxy():
    return AsyncMock()


@pytest.fixture
def mock_label_manager():
    mgr = AsyncMock()
    # get_existing_priority is synchronous — use MagicMock so it returns a value, not a coroutine
    mgr.get_existing_priority = MagicMock(return_value=None)
    return mgr


@pytest.fixture
def cloud_sem():
    return asyncio.Semaphore(2)


@pytest.fixture
def local_sem():
    return asyncio.Semaphore(1)


@pytest.fixture
def sample_headers():
    """Sample Gmail message headers."""
    return [
        {"name": "From", "value": "John Doe <john@example.com>"},
        {"name": "To", "value": "me@example.com"},
        {"name": "Subject", "value": "Meeting tomorrow"},
        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
    ]


@pytest.fixture
def sample_message():
    """Sample full Gmail message resource."""
    body_text = "Hey, can we meet tomorrow at 3pm to discuss the project?"
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    return {
        "id": "msg_001",
        "threadId": "thread_001",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Hey, can we meet tomorrow at 3pm to discuss the project?",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "John Doe <john@example.com>"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": "Meeting tomorrow"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
            ],
            "body": {"data": encoded_body, "size": len(body_text)},
        },
    }


@pytest.fixture
def sample_service_message():
    """Sample service/automated Gmail message resource."""
    body_text = "Your order #12345 has shipped! Track your package at https://example.com/track"
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    return {
        "id": "msg_002",
        "threadId": "thread_002",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Your order #12345 has shipped!",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Amazon <shipment-tracking@amazon.com>"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": "Your Amazon order has shipped"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
            ],
            "body": {"data": encoded_body, "size": len(body_text)},
        },
    }


@pytest.fixture
def sample_labels():
    """Sample Gmail labels list response."""
    return {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
            {"id": "Label_2", "name": "agent/fyi", "type": "user"},
            {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
            {"id": "Label_4", "name": "agent/processed", "type": "user"},
            {"id": "Label_5", "name": "agent/personal", "type": "user"},
            {"id": "Label_6", "name": "agent/non-personal", "type": "user"},
        ]
    }
