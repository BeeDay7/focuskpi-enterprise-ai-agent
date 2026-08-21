from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """
    Application-level authenticated identity.

    This object represents the identity established by the
    authentication boundary. It is intentionally independent
    of the HTTP request body and independent of the
    authorization policy.

    The authorization layer consumes this principal but does
    not determine how authentication was performed.
    """

    id: str
    role: str = "user"


def get_authenticated_principal() -> AuthenticatedPrincipal:
    """
    Development authentication boundary.

    This is a temporary development implementation until the
    application has a real authentication mechanism.

    The returned identity is intentionally NOT derived from
    request.user_id or any tool arguments.
    """

    return AuthenticatedPrincipal(
        id="development-user",
        role="user",
    )