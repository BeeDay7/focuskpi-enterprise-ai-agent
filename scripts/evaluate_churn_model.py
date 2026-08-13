from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from app.db.database import SessionLocal
from app.db.models import Customer
from app.ml.churn import FEATURES, load_model


REPORT_DIR = Path("data") / "reports"


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    predictions = model_result.model.predict(
        X_test
    )

    probabilities = model_result.model.predict_proba(
        X_test
    )[:, 1]

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )

    print("\nChurn Model Evaluation")
    print("======================")

    print(
        f"Accuracy : {model_result.accuracy:.4f}"
    )

    print(
        f"Precision: {model_result.precision:.4f}"
    )

    print(
        f"Recall   : {model_result.recall:.4f}"
    )

    print(
        f"F1       : {model_result.f1:.4f}"
    )

    print(
        f"ROC-AUC  : {model_result.roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix")
    print(matrix)

    print("\nClassification Report")
    print(report)

    report_file = (
        REPORT_DIR / "churn_evaluation.txt"
    )

    report_file.write_text(
        "\n".join(
            [
                "Churn Model Evaluation",
                "======================",
                f"Accuracy : {model_result.accuracy:.4f}",
                f"Precision: {model_result.precision:.4f}",
                f"Recall   : {model_result.recall:.4f}",
                f"F1       : {model_result.f1:.4f}",
                f"ROC-AUC  : {model_result.roc_auc:.4f}",
                f"PR-AUC   : {pr_auc:.4f}",
                "",
                "Confusion Matrix",
                str(matrix),
                "",
                "Classification Report",
                report,
            ]
        ),
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No Churn",
            "Churn",
        ],
    )

    display.plot()

    plt.title(
        "Customer Churn Confusion Matrix"
    )

    chart_file = (
        REPORT_DIR / "churn_confusion_matrix.png"
    )

    plt.savefig(
        chart_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nReport saved to: {report_file}"
    )

    print(
        f"Chart saved to: {chart_file}"
    )


if __name__ == "__main__":
    main()