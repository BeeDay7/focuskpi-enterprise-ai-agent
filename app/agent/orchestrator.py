import re

import pandas as pd
from sqlalchemy.orm import Session

from app.agent.llm_agent import LLMToolAgent
from app.agent.tools import execute_tool
from app.llm.client import LLMError
from app.ml.churn import predict_customer, train_model
from app.services.data_service import (
    all_customers,
    customer_by_code,
    sales_by_region,
)


class Agent:
    """
    Hybrid enterprise agent.

    Uses the real LLM agent when configured. Falls back to deterministic
    execution when live LLM inference is unavailable.
    """

    def __init__(self) -> None:
        self.llm_agent = LLMToolAgent()

    async def run(
        self,
        db: Session,
        message: str,
    ) -> dict:

        if self.llm_agent.client.enabled:

            try:
                return await self.llm_agent.run(
                    db=db,
                    user_message=message,
                )

            except LLMError as exc:
                print(f"LLM ERROR: {exc}")

        return self._demo_run(
            db=db,
            message=message,
        )

    def _demo_run(
        self,
        db: Session,
        message: str,
    ) -> dict:

        text = message.lower()

        calls = []
        data = {}
        response_parts = []

        wants_sales = any(
            keyword in text
            for keyword in (
                "sales",
                "revenue",
                "region",
            )
        )

        wants_churn = any(
            keyword in text
            for keyword in (
                "churn",
                "risk",
                "retention",
            )
        )

        wants_knowledge = any(
            keyword in text
            for keyword in (
                "policy",
                "policies",
                "procedure",
                "procedures",
                "documentation",
                "document",
                "security",
                "api key",
                "secret",
                "support complaints",
                "enterprise",
            )
        )

        # -------------------------------------------------
        # Sales
        # -------------------------------------------------

        if wants_sales:

            result = execute_tool(
                db=db,
                tool_name="get_sales_by_region",
                arguments={},
            )

            data["sales_by_region"] = (
                result.get(
                    "sales_by_region",
                    [],
                )
            )

            calls.append(
                "sales_by_region"
            )

            if data["sales_by_region"]:

                top_region = (
                    data["sales_by_region"][0]
                )

                response_parts.append(
                    f"Top recorded sales region is "
                    f"{top_region['region']} with "
                    f"{top_region['total_sales']:,.2f}."
                )

        # -------------------------------------------------
        # Churn
        # -------------------------------------------------

        if wants_churn:

            customers = all_customers(
                db
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

            model = train_model(
                dataframe
            )

            customer_match = re.search(
                r"(cust-\d+)",
                text,
                re.IGNORECASE,
            )

            if customer_match:

                customer_code = (
                    customer_match
                    .group(1)
                    .upper()
                )

                result = execute_tool(
                    db=db,
                    tool_name="predict_customer_churn",
                    arguments={
                        "customer_code": customer_code
                    },
                )

                calls.append(
                    "predict_customer"
                )

                data["churn_prediction"] = result

                if "error" in result:

                    response_parts.append(
                        result["error"]
                    )

                else:

                    response_parts.append(
                        f"{customer_code} has "
                        f"{result['risk']} churn risk "
                        f"({result['churn_probability']:.1%} "
                        f"estimated probability)."
                    )

            else:

                result = execute_tool(
                    db=db,
                    tool_name="list_highest_churn_risk",
                    arguments={},
                )

                calls.append(
                    "highest_churn_risk"
                )

                data["churn_predictions"] = (
                    result.get(
                        "customers",
                        [],
                    )
                )

                if data["churn_predictions"]:

                    highest = (
                        data["churn_predictions"][0]
                    )

                    response_parts.append(
                        f"Highest modeled churn risk is "
                        f"{highest['customer']} at "
                        f"{highest['churn_probability']:.1%}."
                    )

        # -------------------------------------------------
        # Enterprise knowledge
        # -------------------------------------------------

        if wants_knowledge:

            result = execute_tool(
                db=db,
                tool_name="search_enterprise_knowledge",
                arguments={
                    "query": message,
                    "top_k": 3,
                },
            )

            retrieved = result.get(
                "results",
                [],
            )

            calls.append(
                "search_enterprise_knowledge"
            )

            data["knowledge_results"] = (
                retrieved
            )

            if retrieved:

                top = retrieved[0]

                response_parts.append(
                    "According to the "
                    f"{top['title']}, "
                    f"{top['text']}"
                )

            else:

                response_parts.append(
                    "No relevant enterprise "
                    "knowledge was found."
                )

        # -------------------------------------------------
        # Unsupported
        # -------------------------------------------------

        if not calls:

            return {
                "answer": (
                    "I can currently work with sales "
                    "analytics, customer churn analysis, "
                    "and approved enterprise knowledge."
                ),
                "intent": "unsupported",
                "tool_calls": [],
                "data": {},
                "mode": "demo",
            }

        return {
            "answer": " ".join(
                response_parts
            ),
            "intent": "analytics_ml_and_rag",
            "tool_calls": calls,
            "data": data,
            "mode": "demo",
        }