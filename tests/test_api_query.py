import pytest
from fastapi.testclient import TestClient

from raganything.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_post_query_validation(client):
    resp = client.post("/api/query", json={"query": "", "mode": "hybrid"})
    assert resp.status_code == 422

    resp2 = client.post("/api/query", json={"query": "hello", "mode": "invalid"})
    assert resp2.status_code == 422
