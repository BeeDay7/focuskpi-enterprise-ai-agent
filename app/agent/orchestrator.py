import re

import pandas as pd
from sqlalchemy.orm import Session

from app.agent.llm_agent import LLMToolAgent
from app.agent.tools import execute_tool
from app.llm.client import LLMError
from app.ml.churn import train_model
from app.services.audit_service import write_audit_log
from app.services.data_service import all_customers


class Agent:
    """
    Hybrid enterprise AI agent.

    Uses the real LLM agent when configured and available.
    Falls back to deterministic demo execution when live LLM
    inference is unavailable.

    Authorization is controlled by the application-controlled
    authenticated principal.

    The client/request user_id is not used as the authorization
    principal.

    The public API exposes stable business-level tool names such as:

        sales_by_region
        predict_customer
        highest_churn_risk
        search_enterprise_knowledge
    """

    def __init__(self) -> None:
        self.llm_agent = LLMToolAgent()

    async def run(
        self,
        db: Session,
        message: str,
        user_id: str = "anonymous",
        auth_principal: object | None = None,
    ) -> dict:

        # -------------------------------------------------
        # Live LLM execution
        # -------------------------------------------------

        if self.llm_agent.client.enabled:

            try:
                result = await self.llm_agent.run(
                    db=db,
                    user_message=message,
                    auth_principal=auth_principal,
                )

                result = self._normalize_result(
                    result
                )

                self._write_audit_records(
                    db=db,
                    user_id=user_id,
                    message=message,
                    result=result,
                    success=True,
                )

                return result

            except LLMError as exc:

                print(
                    f"LLM ERROR: {exc}"
                )

                try:
                    fallback_result = self._demo_run(
                        db=db,
                        message=message,
                        auth_principal=auth_principal,
                    )

                    # The user request succeeded through
                    # deterministic fallback. Therefore the
                    # business operation itself is successful.
                    self._write_audit_records(
                        db=db,
                        user_id=user_id,
                        message=message,
                        result=fallback_result,
                        success=True,
                        error_category="llm_error",
                    )

                    return fallback_result

                except Exception as fallback_exc:

                    print(
                        f"FALLBACK ERROR: "
                        f"{fallback_exc}"
                    )

                    error_result = {
                        "answer": (
                            "The AI service and its "
                            "fallback execution were "
                            "unable to process the request."
                        ),
                        "intent": "error",
                        "tool_calls": [],
                        "data": {},
                        "mode": "demo",
                    }

                    self._write_audit_records(
                        db=db,
                        user_id=user_id,
                        message=message,
                        result=error_result,
                        success=False,
                        error_category=(
                            "fallback_error"
                        ),
                    )

                    return error_result

            except Exception as exc:

                print(
                    f"AGENT ERROR: {exc}"
                )

                try:
                    fallback_result = self._demo_run(
                        db=db,
                        message=message,
                        auth_principal=auth_principal,
                    )

                    self._write_audit_records(
                        db=db,
                        user_id=user_id,
                        message=message,
                        result=fallback_result,
                        success=True,
                        error_category="agent_error",
                    )

                    return fallback_result

                except Exception as fallback_exc:

                    print(
                        f"FALLBACK ERROR: "
                        f"{fallback_exc}"
                    )

                    error_result = {
                        "answer": (
                            "The AI agent encountered "
                            "an error and the fallback "
                            "execution also failed."
                        ),
                        "intent": "error",
                        "tool_calls": [],
                        "data": {},
                        "mode": "demo",
                    }

                    self._write_audit_records(
                        db=db,
                        user_id=user_id,
                        message=message,
                        result=error_result,
                        success=False,
                        error_category=(
                            "fallback_error"
                        ),
                    )

                    return error_result

        # -------------------------------------------------
        # Deterministic demo execution
        # -------------------------------------------------

        try:

            result = self._demo_run(
                db=db,
                message=message,
                auth_principal=auth_principal,
            )

            self._write_audit_records(
                db=db,
                user_id=user_id,
                message=message,
                result=result,
                success=True,
            )

            return result

        except Exception as exc:

            print(
                f"DEMO ERROR: {exc}"
            )

            error_result = {
                "answer": (
                    "The request could not be "
                    "processed by the available "
                    "enterprise tools."
                ),
                "intent": "error",
                "tool_calls": [],
                "data": {},
                "mode": "demo",
            }

            self._write_audit_records(
                db=db,
                user_id=user_id,
                message=message,
                result=error_result,
                success=False,
                error_category="demo_error",
            )

            return error_result

    # =====================================================
    # Public contract normalization
    # =====================================================

    @staticmethod
    def _normalize_result(
        result: dict,
    ) -> dict:
        """
        Normalize internal LLM tool names into stable
        public API tool names.

        Internal:

            get_sales_by_region
            predict_customer_churn
            list_highest_churn_risk

        Public:

            sales_by_region
            predict_customer
            highest_churn_risk
        """

        tool_name_map = {
            "get_sales_by_region": (
                "sales_by_region"
            ),
            "predict_customer_churn": (
                "predict_customer"
            ),
            "list_highest_churn_risk": (
                "highest_churn_risk"
            ),
            "search_enterprise_knowledge": (
                "search_enterprise_knowledge"
            ),
        }

        normalized = dict(
            result
        )

        tool_calls = normalized.get(
            "tool_calls",
            [],
        )

        normalized["tool_calls"] = [
            tool_name_map.get(
                tool_name,
                tool_name,
            )
            for tool_name in tool_calls
        ]

        return normalized

    # =====================================================
    # Audit logging
    # =====================================================

    @staticmethod
    def _write_audit_records(
        db: Session,
        user_id: str,
        message: str,
        result: dict,
        success: bool,
        error_category: str | None = None,
    ) -> None:
        """
        Write one audit record for each public tool call.

        `success` describes whether the requested business
        operation ultimately completed successfully.

        `error_category` records an underlying infrastructure
        or execution condition such as:

            llm_error
            agent_error
            fallback_error
            demo_error

        Therefore a successful fallback may legitimately have:

            success=True
            error_category="llm_error"

        Audit logging must never break the primary request.
        """

        tool_calls = result.get(
            "tool_calls",
            [],
        )

        if not tool_calls:
            return

        for tool_name in tool_calls:

            try:

                write_audit_log(
                    db=db,
                    user_id=user_id,
                    operation="chat",
                    tool_name=tool_name,
                    requested_message=message,
                    success=success,
                    error_category=error_category,
                )

            except Exception as exc:

                print(
                    f"AUDIT LOG ERROR: {exc}"
                )

    # =====================================================
    # Deterministic demo agent
    # =====================================================

    def _demo_run(
        self,
        db: Session,
        message: str,
        auth_principal: object | None = None,
    ) -> dict:

        text = message.lower()

        calls = []
        data = {}
        response_parts = []

        # -------------------------------------------------
        # Intent detection
        # -------------------------------------------------

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
        # Sales analytics
        # -------------------------------------------------

        if wants_sales:

            result = execute_tool(
                db=db,
                tool_name="get_sales_by_region",
                arguments={},
                principal=auth_principal,
            )

            if result.get("authorization", {}).get("allowed") is False:
                return {
                    "answer": (
                        "You are not authorized to execute "
                        "the requested enterprise tool."
                    ),
                    "intent": "authorization_denied",
                    "tool_calls": [],
                    "data": {
                        "authorization": result.get(
                            "authorization"
                        )
                    },
                    "mode": "demo",
                }

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
                    data[
                        "sales_by_region"
                    ][0]
                )

                response_parts.append(
                    f"Top recorded sales region is "
                    f"{top_region['region']} with "
                    f"{top_region['total_sales']:,.2f}."
                )

        # -------------------------------------------------
        # Customer churn
        # -------------------------------------------------

        if wants_churn:

            customers = all_customers(
                db
            )

            if not customers:

                response_parts.append(
                    "No customer records are "
                    "available for churn analysis."
                )

            else:

                dataframe = pd.DataFrame(
                    [
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
                            "churned": (
                                customer.churned
                            ),
                        }
                        for customer in customers
                    ]
                )

                # Validate that the current customer
                # dataset can support model training.
                train_model(
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
                        tool_name=(
                            "predict_customer_churn"
                        ),
                        arguments={
                            "customer_code": (
                                customer_code
                            )
                        },
                        principal=auth_principal,
                    )

                    if result.get(
                        "authorization",
                        {},
                    ).get("allowed") is False:
                        return {
                            "answer": (
                                "You are not authorized "
                                "to execute the requested "
                                "enterprise tool."
                            ),
                            "intent": (
                                "authorization_denied"
                            ),
                            "tool_calls": [],
                            "data": {
                                "authorization": result.get(
                                    "authorization"
                                )
                            },
                            "mode": "demo",
                        }

                    calls.append(
                        "predict_customer"
                    )

                    data[
                        "churn_prediction"
                    ] = result

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
                        tool_name=(
                            "list_highest_churn_risk"
                        ),
                        arguments={},
                        principal=auth_principal,
                    )

                    if result.get(
                        "authorization",
                        {},
                    ).get("allowed") is False:
                        return {
                            "answer": (
                                "You are not authorized "
                                "to execute the requested "
                                "enterprise tool."
                            ),
                            "intent": (
                                "authorization_denied"
                            ),
                            "tool_calls": [],
                            "data": {
                                "authorization": result.get(
                                    "authorization"
                                )
                            },
                            "mode": "demo",
                        }

                    calls.append(
                        "highest_churn_risk"
                    )

                    data[
                        "churn_predictions"
                    ] = result.get(
                        "customers",
                        [],
                    )

                    if data[
                        "churn_predictions"
                    ]:

                        highest = (
                            data[
                                "churn_predictions"
                            ][0]
                        )

                        response_parts.append(
                            f"Highest modeled churn risk "
                            f"is {highest['customer']} "
                            f"at "
                            f"{highest['churn_probability']:.1%}."
                        )

        # -------------------------------------------------
        # Enterprise knowledge / RAG
        # -------------------------------------------------

        if wants_knowledge:

            result = execute_tool(
                db=db,
                tool_name=(
                    "search_enterprise_knowledge"
                ),
                arguments={
                    "query": message,
                    "top_k": 3,
                },
                principal=auth_principal,
            )

            if result.get(
                "authorization",
                {},
            ).get("allowed") is False:
                return {
                    "answer": (
                        "You are not authorized "
                        "to execute the requested "
                        "enterprise tool."
                    ),
                    "intent": (
                        "authorization_denied"
                    ),
                    "tool_calls": [],
                    "data": {
                        "authorization": result.get(
                            "authorization"
                        )
                    },
                    "mode": "demo",
                }

            retrieved = result.get(
                "results",
                [],
            )

            calls.append(
                "search_enterprise_knowledge"
            )

            data[
                "knowledge_results"
            ] = retrieved

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
        # Unsupported request
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

        # -------------------------------------------------
        # Successful analytics / ML / RAG response
        # -------------------------------------------------

        return {
            "answer": " ".join(
                response_parts
            ),
            "intent": "analytics_ml_and_rag",
            "tool_calls": calls,
            "data": data,
            "mode": "demo",
        }