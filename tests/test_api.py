import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ailearn.api import create_app  # noqa: E402


@pytest.fixture
def client_and_log():
    executed = []
    app = create_app(action_executor=lambda a: executed.append(a) or f"Executed: {a}")
    return TestClient(app), executed


def test_health(client_and_log):
    client, _ = client_and_log
    assert client.get("/health").json() == {"status": "ok"}


def test_sql_question_answers_directly(client_and_log):
    client, _ = client_and_log
    r = client.post("/ask", json={"question": "What is total revenue by region?"}).json()
    assert r["status"] == "done" and "north" in r["answer"]
    assert any(t.startswith("run_sql") for t in r["trace"])


def test_action_requires_approval(client_and_log):
    client, executed = client_and_log
    r = client.post("/ask", json={"question": "rerun the sales ingestion pipeline", "thread_id": "x1"}).json()
    assert r["status"] == "pending_approval" and executed == []
    r = client.post("/approve/x1", json={"approved": True}).json()
    assert r["status"] == "done" and len(executed) == 1


def test_reject_and_double_approve(client_and_log):
    client, executed = client_and_log
    client.post("/ask", json={"question": "delete old partitions", "thread_id": "x2"})
    r = client.post("/approve/x2", json={"approved": False}).json()
    assert "NOT executed" in r["answer"] and executed == []
    assert client.post("/approve/x2", json={"approved": True}).status_code == 409


def test_validation_and_unknown_thread(client_and_log):
    client, _ = client_and_log
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.get("/threads/nope").status_code == 404
