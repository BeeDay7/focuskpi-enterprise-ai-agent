from types import SimpleNamespace

from app.agent.tools import execute_tool


def test_execute_tool_allows_get_sales_by_region(monkeypatch):
    called = []

    def fake_sales(db):
        called.append(True)
        return [{"region": "North", "total_sales": 100.0}]

    monkeypatch.setattr("app.agent.tools.sales_by_region", fake_sales)

    result = execute_tool(db="session", tool_name="get_sales_by_region", arguments={}, principal="user-1")

    assert called == [True]
    assert "sales_by_region" in result
    assert result["sales_by_region"][0]["region"] == "North"


def test_execute_tool_allows_predict_customer_churn(monkeypatch):
    # Provide a fake customer and stub out model/prediction to keep test fast
    def fake_customer_by_code(db, code):
        return SimpleNamespace(
            customer_code=code,
            tenure_months=12,
            monthly_spend=100.0,
            support_tickets=0,
            late_payments=0,
            churned=False,
        )

    monkeypatch.setattr("app.agent.tools.customer_by_code", fake_customer_by_code)
    monkeypatch.setattr("app.agent.tools._train_churn_model", lambda db: "model")

    def fake_predict(model, features):
        return {
            "churn_probability": 0.42,
            "risk": "medium",
            "model_accuracy": 0.9,
            "model_precision": 0.8,
            "model_recall": 0.7,
            "model_f1": 0.75,
            "model_roc_auc": 0.85,
        }

    monkeypatch.setattr("app.agent.tools.predict_customer", fake_predict)

    result = execute_tool(
        db="session",
        tool_name="predict_customer_churn",
        arguments={"customer_code": "CUST-1007"},
        principal="user-2",
    )

    assert result["customer"] == "CUST-1007"
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk"] in {"low", "medium", "high"}


def test_execute_tool_allows_list_highest_churn_risk(monkeypatch):
    # Return two fake customers
    def fake_all_customers(db):
        return [
            SimpleNamespace(
                customer_code="CUST-1",
                tenure_months=10,
                monthly_spend=100,
                support_tickets=1,
                late_payments=0,
                churned=False,
            ),
            SimpleNamespace(
                customer_code="CUST-2",
                tenure_months=5,
                monthly_spend=200,
                support_tickets=2,
                late_payments=1,
                churned=False,
            ),
        ]

    monkeypatch.setattr("app.agent.tools.all_customers", fake_all_customers)
    monkeypatch.setattr("app.agent.tools._train_churn_model", lambda db: "model")
    monkeypatch.setattr("app.agent.tools.predict_customer", lambda model, f: {"churn_probability": 0.5, "risk": "medium"})

    result = execute_tool(db="session", tool_name="list_highest_churn_risk", arguments={}, principal="u3")

    assert "customers" in result
    assert isinstance(result["customers"], list)


def test_execute_tool_allows_search_enterprise_knowledge(monkeypatch):
    # Stub RAG retriever
    monkeypatch.setattr("app.agent.tools.RAG_RETRIEVER", SimpleNamespace(search=lambda **kwargs: [{"title": "Security Policy", "text": "Do X", "document_id": "SECURITY_POLICY", "score": 0.9}] ))

    result = execute_tool(
        db="session",
        tool_name="search_enterprise_knowledge",
        arguments={"query": "security policy", "top_k": 1},
        principal="alice",
    )

    assert "results" in result or "results" in result or "query" in result
    # results may be under 'results' key
    if "results" in result:
        assert len(result["results"]) > 0


def test_missing_principal_denied_and_tool_not_executed(monkeypatch):
    def fail_if_called(db):
        raise RuntimeError("should not be called")

    monkeypatch.setattr("app.agent.tools.sales_by_region", fail_if_called)

    result = execute_tool(db="session", tool_name="get_sales_by_region", arguments={}, principal=None)

    assert result.get("error") == "Tool execution denied."
    assert result["authorization"]["reason_code"] == "missing_principal"


def test_anonymous_principal_denied(monkeypatch):
    def fail_if_called(db):
        raise RuntimeError("should not be called")

    monkeypatch.setattr("app.agent.tools.sales_by_region", fail_if_called)

    result = execute_tool(db="session", tool_name="get_sales_by_region", arguments={}, principal="anonymous")

    assert result.get("error") == "Tool execution denied."
    assert result["authorization"]["reason_code"] == "missing_principal"


def test_unknown_tool_denied():
    result = execute_tool(db="session", tool_name="not_registered", arguments={}, principal="user-1")

    assert result.get("error") == "Tool execution denied."
    assert result["authorization"]["reason_code"] == "unknown_tool"


def test_tool_arguments_cannot_override_principal(monkeypatch):
    # Even if arguments contain a user_id, the principal param controls auth
    def fail_if_called(db):
        raise RuntimeError("should not be called")

    monkeypatch.setattr("app.agent.tools.sales_by_region", fail_if_called)

    result = execute_tool(db="session", tool_name="get_sales_by_region", arguments={"user_id": "attacker"}, principal=None)

    assert result.get("error") == "Tool execution denied."
    assert result["authorization"]["reason_code"] == "missing_principal"


def test_tool_name_is_normalized_before_authorization_and_execution(monkeypatch):
    called = []

    def fake_sales(db):
        called.append(True)
        return [{"region": "North", "total_sales": 100.0}]

    monkeypatch.setattr("app.agent.tools.sales_by_region", fake_sales)

    result = execute_tool(
        db="session",
        tool_name="  get_sales_by_region  ",
        arguments={},
        principal="user-1",
    )

    assert called == [True]
    assert result["sales_by_region"][0]["region"] == "North"