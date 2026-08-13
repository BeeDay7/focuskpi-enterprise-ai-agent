from app.security.authorization import (
    AuthorizationDecision,
    ToolAuthorizationPolicy,
    authorize_tool,
)


def test_allowed_tool_for_valid_principal():
    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="get_sales_by_region",
    )

    assert isinstance(decision, AuthorizationDecision)
    assert decision.allowed is True
    assert decision.decision == "ALLOW"
    assert decision.reason_code == "allowed_by_policy"


def test_unknown_tool_denied_for_valid_principal():
    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="unknown_tool",
    )

    assert decision.allowed is False
    assert decision.decision == "DENY"
    assert decision.reason_code == "unknown_tool"


def test_missing_principal_denied():
    decision = authorize_tool(
        principal=None,
        tool_name="get_sales_by_region",
    )

    assert decision.allowed is False
    assert decision.reason_code == "missing_principal"


def test_invalid_principal_denied():
    decision = authorize_tool(
        principal={"user_id": ""},
        tool_name="get_sales_by_region",
    )

    assert decision.allowed is False
    assert decision.reason_code == "missing_principal"

    decision = authorize_tool(
        principal="anonymous",
        tool_name="get_sales_by_region",
    )

    assert decision.allowed is False
    assert decision.reason_code == "missing_principal"


def test_explicitly_denied_tool_denied():
    policy = ToolAuthorizationPolicy(
        allowed_tools={"get_sales_by_region"},
        denied_tools={"get_sales_by_region"},
    )

    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="get_sales_by_region",
        policy=policy,
    )

    assert decision.allowed is False
    assert decision.reason_code == "tool_explicitly_denied"


def test_empty_or_invalid_tool_name_denied():
    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="   ",
    )

    assert decision.allowed is False
    assert decision.reason_code == "invalid_tool_name"

    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name=None,
    )

    assert decision.allowed is False
    assert decision.reason_code == "invalid_tool_name"


def test_authorization_not_affected_by_tool_arguments():
    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="get_sales_by_region",
        context={
            "arguments": {"user_id": "attacker-user"},
            "tool_call": {"user_id": "attacker-user"},
        },
    )

    assert decision.allowed is True
    assert decision.principal_id == "user-123"
    assert decision.decision == "ALLOW"


def test_user_id_in_arguments_cannot_override_authenticated_principal():
    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="get_sales_by_region",
        context={
            "arguments": {"user_id": "user-999"},
            "metadata": {"current_user": "user-999"},
        },
    )

    assert decision.principal_id == "user-123"
    assert decision.allowed is True


def test_unknown_tools_never_become_authorized_implicitly():
    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="not_in_policy",
    )

    assert decision.allowed is False
    assert decision.reason_code == "unknown_tool"


def test_authz_is_deterministic_for_same_inputs():
    first = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="search_enterprise_knowledge",
    )
    second = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="search_enterprise_knowledge",
    )

    assert first == second
    assert first.allowed is True


def test_unexpected_policy_conditions_fail_closed():
    class BrokenPolicy(ToolAuthorizationPolicy):
        def evaluate(self, principal, tool_name, context=None):
            raise RuntimeError("simulated policy failure")

    decision = authorize_tool(
        principal={"user_id": "user-123"},
        tool_name="get_sales_by_region",
        policy=BrokenPolicy(),
    )

    assert decision.allowed is False
    assert decision.reason_code == "policy_error"
    assert decision.decision == "DENY"
