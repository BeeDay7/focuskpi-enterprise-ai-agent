from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.db.database import SessionLocal
from app.db.models import Customer
from app.ml.churn import FEATURES, load_model


REPORT_DIR = Path("data") / "reports"


def main() -> None:
    model_result = load_model()

    if model_result is None:
        raise RuntimeError(
            "No trained model found. "
            "Run: python -m scripts.train_churn_model"
        )

    db = SessionLocal()

    try:
        customers = db.query(Customer).all()

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
    finally:
        db.close()

    X = dataframe[FEATURES]
    y = dataframe["churned"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    probabilities = model_result.model.predict_proba(
        X_test
    )[:, 1]

    rows = []

    for threshold in [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": round(
                    precision,
                    4,
                ),
                "recall": round(
                    recall,
                    4,
                ),
                "f1": round(
                    f1,
                    4,
                ),
            }
        )

    results = pd.DataFrame(rows)

    best = results.loc[
        results["f1"].idxmax()
    ]

    print("\nChurn Threshold Evaluation")
    print("==========================")
    print(results.to_string(index=False))

    print("\nBest F1 threshold")
    print("=================")
    print(
        f"Threshold: {best['threshold']:.2f}"
    )
    print(
        f"Precision: {best['precision']:.4f}"
    )
    print(
        f"Recall:    {best['recall']:.4f}"
    )
    print(
        f"F1:        {best['f1']:.4f}"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        REPORT_DIR / "threshold_evaluation.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()