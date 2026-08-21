from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_ALLOWED_TOOLS = frozenset(
    {
        "get_sales_by_region",
        "predict_customer_churn",
        "list_highest_churn_risk",
        "search_enterprise_knowledge",
    }
)


def _normalize_tool_name(tool_name: Any) -> str:
    if tool_name is None:
        return ""

    tool_name = str(tool_name).strip()

    if not tool_name:
        return ""

    return tool_name


def _extract_principal_id(principal: Any) -> str:
    if principal is None:
        return ""

    if isinstance(principal, str):
        return principal.strip()

    if isinstance(principal, Mapping):
        for key in (
            "user_id",
            "principal_id",
            "subject",
            "username",
            "id",
            "email",
        ):
            value = principal.get(key)
            if value is not None:
                return str(value).strip()
        return ""

    for attr in (
        "user_id",
        "principal_id",
        "subject",
        "username",
        "id",
        "email",
    ):
        value = getattr(principal, attr, None)
        if value is not None:
            return str(value).strip()

    return ""


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Structured authorization outcome suitable for later audit logging."""

    allowed: bool
    principal_id: str | None
    tool_name: str
    decision: str
    reason_code: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_audit_payload(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "decision": self.decision,
            "principal_id": self.principal_id,
            "tool_name": self.tool_name,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "context": dict(self.context),
        }


class ToolAuthorizationPolicy:
    """Deterministic whitelist policy with deny-by-default semantics."""

    def __init__(
        self,
        allowed_tools: set[str] | frozenset[str] | None = None,
        denied_tools: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.allowed_tools = (
            frozenset(DEFAULT_ALLOWED_TOOLS)
            if allowed_tools is None
            else frozenset(allowed_tools)
        )

        self.denied_tools = (
            frozenset()
            if denied_tools is None
            else frozenset(denied_tools)
        )

    def evaluate(
        self,
        principal: Any,
        tool_name: Any,
        context: Mapping[str, Any] | None = None,
    ) -> AuthorizationDecision:
        safe_context = dict(context or {})

        try:
            principal_id = _extract_principal_id(principal)
            normalized_tool = _normalize_tool_name(tool_name)

            safe_context = {
                key: value
                for key, value in safe_context.items()
                if key
                not in {
                    "password",
                    "token",
                    "secret",
                    "credentials",
                }
            }

            if not principal_id or principal_id.lower() == "anonymous":
                return AuthorizationDecision(
                    allowed=False,
                    principal_id=principal_id or None,
                    tool_name=normalized_tool,
                    decision="DENY",
                    reason_code="missing_principal",
                    reason="Authenticated principal is missing or invalid.",
                    context=safe_context,
                )

            if not normalized_tool:
                return AuthorizationDecision(
                    allowed=False,
                    principal_id=principal_id,
                    tool_name="",
                    decision="DENY",
                    reason_code="invalid_tool_name",
                    reason="Tool name is missing or invalid.",
                    context=safe_context,
                )

            if normalized_tool in self.denied_tools:
                return AuthorizationDecision(
                    allowed=False,
                    principal_id=principal_id,
                    tool_name=normalized_tool,
                    decision="DENY",
                    reason_code="tool_explicitly_denied",
                    reason="Tool is explicitly denied by policy.",
                    context=safe_context,
                )

            if normalized_tool not in self.allowed_tools:
                return AuthorizationDecision(
                    allowed=False,
                    principal_id=principal_id,
                    tool_name=normalized_tool,
                    decision="DENY",
                    reason_code="unknown_tool",
                    reason="Tool is not explicitly allowed by policy.",
                    context=safe_context,
                )

            if not self.allowed_tools:
                return AuthorizationDecision(
                    allowed=False,
                    principal_id=principal_id,
                    tool_name=normalized_tool,
                    decision="DENY",
                    reason_code="policy_error",
                    reason="Policy is empty or invalid.",
                    context=safe_context,
                )

            return AuthorizationDecision(
                allowed=True,
                principal_id=principal_id,
                tool_name=normalized_tool,
                decision="ALLOW",
                reason_code="allowed_by_policy",
                reason="Tool is explicitly allowed for the authenticated principal.",
                context=safe_context,
            )

        except Exception:
            return AuthorizationDecision(
                allowed=False,
                principal_id=_extract_principal_id(principal) or None,
                tool_name=_normalize_tool_name(tool_name),
                decision="DENY",
                reason_code="policy_error",
                reason="Unexpected policy condition encountered; fail-closed denied access.",
                context=safe_context,
            )


class ToolAuthorizationService:
    def __init__(
        self,
        policy: ToolAuthorizationPolicy | None = None,
    ) -> None:
        self.policy = policy or ToolAuthorizationPolicy()

    def authorize(
        self,
        principal: Any,
        tool_name: Any,
        context: Mapping[str, Any] | None = None,
    ) -> AuthorizationDecision:
        try:
            return self.policy.evaluate(
                principal=principal,
                tool_name=tool_name,
                context=context,
            )

        except Exception:
            return AuthorizationDecision(
                allowed=False,
                principal_id=_extract_principal_id(principal) or None,
                tool_name=_normalize_tool_name(tool_name),
                decision="DENY",
                reason_code="policy_error",
                reason="Unexpected policy condition encountered; fail-closed denied access.",
                context=dict(context or {}),
            )


def authorize_tool(
    principal: Any,
    tool_name: Any,
    context: Mapping[str, Any] | None = None,
    policy: ToolAuthorizationPolicy | None = None,
) -> AuthorizationDecision:
    service = ToolAuthorizationService(policy=policy)

    return service.authorize(
        principal=principal,
        tool_name=tool_name,
        context=context,
    )