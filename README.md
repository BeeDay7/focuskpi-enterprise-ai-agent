# Enterprise AI Data Intelligence Agent

Portfolio project for AI/ML engineering applications. Demonstrates a Python/FastAPI agent architecture that connects conversational requests to enterprise-style SQL analytics and machine-learning prediction.

## Features
- Controlled agent/tool orchestration
- SQL/database integration
- PostgreSQL-ready configuration
- FastAPI REST API
- Random Forest churn prediction
- pandas/scikit-learn preprocessing and evaluation
- Interactive Swagger API docs
- Automated tests
- LLM-ready architecture

## Quick start

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_demo.py
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs`.

Try:
```json
{"message":"Show sales by region"}
```

or:
```json
{"message":"Predict churn risk for CUST-1007"}
```

## Architecture

```text
Client -> FastAPI -> Agent Orchestrator -> Controlled Tools
                                      |-> SQL Database
                                      |-> ML Model
                                      -> Structured Response
```

## Roadmap
- Phase 2: production LLM provider adapter
- Phase 3: RAG + embeddings/vector store
- Phase 4: PostgreSQL + migrations
- Phase 5: authentication, authorization, audit logging
- Phase 6: Docker + cloud deployment
- Phase 7: evaluation, observability, CI/CD
