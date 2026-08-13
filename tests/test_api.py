from fastapi.testclient import TestClient

from app.main import app
from app.db.database import Base, engine

from scripts.seed_demo import main as seed_database


Base.metadata.create_all(bind=engine)

client = TestClient(app)


def setup_module():
    """
    Rebuild the enterprise test database before the test module runs.
    This keeps tests aligned with the same synthetic dataset used by
    local development and model training.
    """
    seed_database()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_sales():
    response = client.post(
        "/api/chat",
        json={
            "message": "show sales by region"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "sales_by_region" in body["tool_calls"]

    assert "sales_by_region" in body["data"]

    assert len(
        body["data"]["sales_by_region"]
    ) > 1


def test_churn():
    response = client.post(
        "/api/chat",
        json={
            "message": "predict churn risk for CUST-1007"
        },
    )

    assert response.status_code == 200

    body = response.json()

    prediction = body["data"]["churn_prediction"]

    assert prediction["customer"] == "CUST-1007"

    assert 0.0 <= prediction["churn_probability"] <= 1.0

    assert prediction["risk"] in {
        "low",
        "medium",
        "high",
    }

    assert "model_accuracy" in prediction

    assert "model_precision" in prediction

    assert "model_recall" in prediction

    assert "model_f1" in prediction

    assert "model_roc_auc" in prediction


def test_enterprise_knowledge():
    response = client.post(
        "/api/chat",
        json={
            "message": "What is our security policy?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["mode"] == "demo"

    assert (
        "search_enterprise_knowledge"
        in body["tool_calls"]
    )

    assert "knowledge_results" in body["data"]

    assert len(
        body["data"]["knowledge_results"]
    ) > 0

    first_result = (
        body["data"]["knowledge_results"][0]
    )

    assert first_result["title"] == "Security Policy"

    assert (
        first_result["document_id"]
        == "SECURITY_POLICY"
    )


def test_unsupported_request():
    response = client.post(
        "/api/chat",
        json={
            "message": "What is the weather today?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["intent"] == "unsupported"

    assert body["tool_calls"] == []

    assert body["data"] == {}

    assert body["mode"] == "demo"


def test_llm_failure_falls_back_to_demo(monkeypatch):
    from app.llm.client import LLMError

    def fake_chat(*args, **kwargs):
        raise LLMError(
            "Simulated LLM failure"
        )

    monkeypatch.setattr(
        "app.llm.client.LLMClient.chat",
        fake_chat,
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "show sales by region"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["mode"] == "demo"

    assert (
        "sales_by_region"
        in body["tool_calls"]
    )

    assert (
        "sales_by_region"
        in body["data"]
    )