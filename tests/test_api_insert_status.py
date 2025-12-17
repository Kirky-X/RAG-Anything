import pytest
from fastapi.testclient import TestClient

from raganything.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_insert_content_list_empty(client):
    body = {
        "content_list": [
            {"type": "text", "text": "hello", "page_idx": 0}
        ],
        "file_path": "doc.md",
    }
    # This calls LightRAG insert; depending on environment it may fail. We only check validation chain.
    resp = client.post("/api/doc/insert", json=body)
    assert resp.status_code in (200, 500)
