from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.orchestrator import Agent
from app.db.database import SessionLocal
from app.schemas.chat import ChatRequest, ChatResponse


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
    )