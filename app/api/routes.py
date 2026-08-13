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


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    return await agent.run(
        db=db,
        message=request.message,
        user_id=request.user_id,
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