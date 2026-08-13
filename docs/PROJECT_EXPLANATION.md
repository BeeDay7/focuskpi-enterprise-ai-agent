# FocusKPI Project Explanation

## Problem
Enterprise users often need information from databases and predictive models but may not know SQL or how to invoke ML services. This project provides a conversational interface for controlled analytics and ML inference.

## Architecture and tools
The backend uses Python/FastAPI, SQLAlchemy, SQLite for a runnable demo and PostgreSQL-ready configuration. pandas and scikit-learn provide the ML workflow. The agent orchestrates controlled tools rather than allowing arbitrary model-generated database execution. An LLM can be placed above the orchestration layer later for natural-language intent routing.

## Contribution
I designed the architecture and implemented the API, database models, controlled tool layer, ML workflow, and agent orchestration. The design separates reasoning from execution and keeps the database and model operations behind explicit application functions.

## Challenges and solutions
The central challenge was preserving the flexibility of an AI agent while maintaining reliable execution. I addressed this with a tool boundary: the agent selects predefined operations, while application code performs database and ML actions. This also makes the system easier to test, monitor, secure, and extend with enterprise APIs, RAG, and LLM providers.
