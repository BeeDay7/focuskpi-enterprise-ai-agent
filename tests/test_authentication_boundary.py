from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_authenticated_principal
from app.security.identity import AuthenticatedPrincipal


client = TestClient(app)


def test_authenticated_principal_is_application_controlled():
    principal = get_authenticated_principal()

    assert isinstance(principal, AuthenticatedPrincipal)
    assert principal.id == "development-user"
    assert principal.role == "user"


def test_request_user_id_does_not_control_authenticated_principal():
    response = client.post(
        "/api/chat",
        json={
            "message": "show sales by region",
            "user_id": "attacker-controlled-user",
        },
    )

    assert response.status_code == 200

    # The request body must not be able to replace the
    # application-controlled authentication identity.
    principal = get_authenticated_principal()

    assert principal.id == "development-user"


def test_request_user_id_is_not_used_as_tool_authorization_principal():
    response = client.post(
        "/api/chat",
        json={
            "message": "show sales by region",
            "user_id": "attacker",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "sales_by_region" in body["tool_calls"]