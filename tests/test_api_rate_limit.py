import asyncio
import json

import pytest

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

from cricketwp.api import main as api
from cricketwp.api.main import check_rate_limit, rate_limit_store


client = TestClient(api.app)


def test_rate_limit_blocks_61st_request():
    client_ip = "test-client-61"

    try:
        # First 60 requests should succeed.
        for _ in range(60):
            check_rate_limit(client_ip)

        # 61st request should be rejected.
        with pytest.raises(HTTPException) as exc:
            check_rate_limit(client_ip)

        assert exc.value.status_code == 429
        assert "Retry-After" in exc.value.headers

    finally:
        rate_limit_store.pop(client_ip, None)


def test_rate_limit_resets_after_window(monkeypatch):
    client_ip = "test-client-reset"
    current_time = [1000.0]

    monkeypatch.setattr(
        api.time,
        "monotonic",
        lambda: current_time[0],
    )

    try:
        # Use all 60 requests.
        for _ in range(60):
            api.check_rate_limit(client_ip)

        # Advance time by 60 seconds.
        current_time[0] += 60

        # A new request should now succeed.
        api.check_rate_limit(client_ip)

    finally:
        api.rate_limit_store.pop(client_ip, None)


def test_http_error_uses_standard_response():
    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404

    body = response.json()

    assert body["error"]["code"] == "http_error"
    assert body["error"]["message"]
    assert body["request_id"]
    assert response.headers["x-request-id"] == body["request_id"]


def test_validation_error_uses_standard_response():
    response = client.post(
        "/api/v1/predict/chase",
        json={},
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Request validation failed."
    assert isinstance(body["error"]["details"], list)
    assert body["request_id"]


def test_http_429_uses_standard_response():
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/predict/chase",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 12345),
            "scheme": "http",
            "state": {"request_id": "test-request-123"},
        }
    )

    exc = StarletteHTTPException(
        status_code=429,
        detail="Rate limit exceeded. Try again later.",
        headers={"Retry-After": "60"},
    )

    response = asyncio.run(api.http_exception_handler(request, exc))

    body = json.loads(response.body)

    assert response.status_code == 429
    assert body["error"]["code"] == "http_error"
    assert body["error"]["message"] == ("Rate limit exceeded. Try again later.")
    assert body["request_id"] == "test-request-123"
    assert response.headers["Retry-After"] == "60"


def test_internal_error_uses_safe_standard_response():
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 12345),
            "scheme": "http",
            "state": {"request_id": "test-500-request"},
        }
    )

    exc = RuntimeError("Sensitive internal database error")

    response = asyncio.run(api.unhandled_exception_handler(request, exc))

    body = json.loads(response.body)

    assert response.status_code == 500
    assert body["error"]["code"] == "internal_server_error"
    assert body["error"]["message"] == "An unexpected error occurred."
    assert body["request_id"] == "test-500-request"
    assert "Sensitive internal database error" not in response.body.decode()
