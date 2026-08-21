from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.ml.churn import predict_customer, train_model
from app.rag.retriever import LocalRetriever
from app.services.data_service import (
    all_customers,
    customer_by_code,
    sales_by_region,
)


# ---------------------------------------------------------
# Enterprise knowledge retriever
# ---------------------------------------------------------

RAG_RETRIEVER = LocalRetriever(
    knowledge_base_path="data/knowledge_base"
)


# ---------------------------------------------------------
# LLM tool definitions
# ---------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_by_region",
            "description": (
                "Return total recorded sales grouped by business region."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_customer_churn",
            "description": (
                "Predict the churn risk of a specific customer "
                "using the trained machine-learning model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_code": {
                        "type": "string",
                        "description": (
                            "Customer identifier such as CUST-1007."
                        ),
                    }
                },
                "required": [
                    "customer_code"
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_highest_churn_risk",
            "description": (
                "Return the five customers with the highest "
                "predicted churn probability."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_enterprise_knowledge",
            "description": (
                "Search approved enterprise knowledge documents "
                "and return the most relevant passages for a question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The enterprise knowledge question "
                            "to search for."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Number of relevant passages to return."
                        ),
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------
# ML helper
# ---------------------------------------------------------

def _train_churn_model(db: Session):
    """
    Build the churn model from the current enterprise customer
    dataset.
    """

    customers = all_customers(db)

    if not customers:
        raise RuntimeError(
            "No customers are available for churn-model training."
        )

    dataframe = pd.DataFrame(
        [
            {
                "tenure_months": customer.tenure_months,
                "monthly_spend": customer.monthly_spend,
                "support_tickets": customer.support_tickets,
                "late_payments": customer.late_payments,
                "churned": customer.churned,
            }
            for customer in customers
        ]
    )

    return train_model(
        dataframe
    )


# ---------------------------------------------------------
# Tool execution
# ---------------------------------------------------------

from app.security.authorization import authorize_tool


def execute_tool(
    db: Session,
    tool_name: str,
    arguments: dict[str, Any],
    principal: Any = None,
) -> dict[str, Any]:

    # Evaluate authorization before doing any work
    safe_arguments = dict(arguments or {})

    decision = authorize_tool(
        principal=principal,
        tool_name=tool_name,
        context={"arguments": safe_arguments},
    )

    if not decision.allowed:
        # Deterministic denial response without leaking secrets
        return {
            "error": "Tool execution denied.",
            "authorization": {
                "allowed": False,
                "decision": decision.decision,
                "reason_code": decision.reason_code,
                "reason": decision.reason,
                "principal_id": decision.principal_id,
                "tool_name": decision.tool_name,
            },
        }

    # -----------------------------------------------------
    # Sales analytics
    # -----------------------------------------------------

    if tool_name == "get_sales_by_region":

        return {
            "sales_by_region": sales_by_region(
                db
            )
        }

    # -----------------------------------------------------
    # Customer churn prediction
    # -----------------------------------------------------

    if tool_name == "predict_customer_churn":

        customer_code = str(
            arguments["customer_code"]
        ).strip().upper()

        customer = customer_by_code(
            db,
            customer_code,
        )

        if customer is None:
            return {
                "error": (
                    f"Customer {customer_code} "
                    "was not found."
                )
            }

        model = _train_churn_model(
            db
        )

        prediction = predict_customer(
            model,
            {
                "tenure_months": (
                    customer.tenure_months
                ),
                "monthly_spend": (
                    customer.monthly_spend
                ),
                "support_tickets": (
                    customer.support_tickets
                ),
                "late_payments": (
                    customer.late_payments
                ),
            },
        )

        return {
            "customer": customer_code,
            **prediction,
        }

    # -----------------------------------------------------
    # Highest churn risk ranking
    # -----------------------------------------------------

    if tool_name == "list_highest_churn_risk":

        customers = all_customers(
            db
        )

        if not customers:
            return {
                "customers": []
            }

        model = _train_churn_model(
            db
        )

        predictions = []

        for customer in customers:

            prediction = predict_customer(
                model,
                {
                    "tenure_months": (
                        customer.tenure_months
                    ),
                    "monthly_spend": (
                        customer.monthly_spend
                    ),
                    "support_tickets": (
                        customer.support_tickets
                    ),
                    "late_payments": (
                        customer.late_payments
                    ),
                },
            )

            predictions.append(
                {
                    "customer": (
                        customer.customer_code
                    ),
                    **prediction,
                }
            )

        predictions.sort(
            key=lambda item: item[
                "churn_probability"
            ],
            reverse=True,
        )

        return {
            "customers": predictions[:5]
        }

    # -----------------------------------------------------
    # Enterprise knowledge retrieval
    # -----------------------------------------------------

    if tool_name == "search_enterprise_knowledge":

        query = str(
            arguments["query"]
        ).strip()

        if not query:
            return {
                "query": query,
                "results": [],
            }

        requested_top_k = arguments.get(
            "top_k",
            3,
        )

        try:
            top_k = int(
                requested_top_k
            )
        except (
            TypeError,
            ValueError,
        ):
            top_k = 3

        top_k = max(
            1,
            min(
                top_k,
                5,
            ),
        )

        results = RAG_RETRIEVER.search(
            query=query,
            top_k=top_k,
        )

        return {
            "query": query,
            "results": results,
        }

    # -----------------------------------------------------
    # Unknown tool
    # -----------------------------------------------------

    raise ValueError(
        f"Tool '{tool_name}' is not registered."
    )