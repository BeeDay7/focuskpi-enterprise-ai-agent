from dataclasses import dataclass

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.orchestrator import Agent
from app.db.database import SessionLocal
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.audit_service import recent_audit_logs


router = APIRouter(
    prefix="/api",
    tags=["AI Agent"],
)

agent = Agent()


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """
    Temporary development authenticated principal.

    This is intentionally separate from request.user_id.

    user_id:
        Used for business/audit identity.

    auth_principal:
        Used by the authorization enforcement boundary.

    This development principal must eventually be replaced
    by the application's real authentication middleware.
    """

    id: str
    role: str = "user"


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def get_authenticated_principal() -> AuthenticatedPrincipal:
    """
    Temporary development authentication boundary.

    The current Phase 2 application does not yet have a
    production authentication middleware, so we provide
    an explicit authenticated principal here.

    The request body does NOT control this identity.
    """

    return AuthenticatedPrincipal(
        id="development-user",
        role="user",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    auth_principal: AuthenticatedPrincipal = Depends(
        get_authenticated_principal
    ),
):
    return await agent.run(
        db=db,
        message=request.message,
        user_id=request.user_id,
        auth_principal=auth_principal,
    )


@router.get(
    "/audit",
)
def audit_logs(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    logs = recent_audit_logs(
        db=db,
        limit=limit,
    )

    return {
        "count": len(logs),
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "operation": log.operation,
                "tool_name": log.tool_name,
                "requested_message": log.requested_message,
                "timestamp": log.timestamp.isoformat(),
                "success": log.success,
                "error_category": log.error_category,
            }
            for log in logs
        ],
    }