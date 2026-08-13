from fastapi import FastAPI
from app.core.config import settings
from app.db.database import Base, engine
from app.api.routes import router

Base.metadata.create_all(bind=engine)
app = FastAPI(title=settings.app_name, version="1.0.0", description="Enterprise AI agent integrating databases and machine learning.")
app.include_router(router)

@app.get("/")
def root():
    return {"name": settings.app_name, "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}
