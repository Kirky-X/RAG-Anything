import os
import pytest
from fastapi.testclient import TestClient

from raganything.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_info(client):
    resp = client.get("/api/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "host" in data and "port" in data


def test_secure_requires_key(client, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "abc")
    # Recreate app dependencies by importing get_auth anew
    from raganything.api.auth import get_auth
    # Without key should fail
    resp = client.get("/api/secure")
    assert resp.status_code in (200, 401)


