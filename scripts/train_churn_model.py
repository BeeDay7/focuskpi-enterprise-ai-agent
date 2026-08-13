import pandas as pd
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Customer
from app.ml.churn import train_model


def main() -> None:

    db = SessionLocal()

    try:

        customers = db.execute(
            select(Customer)
        ).scalars().all()

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

        result = train_model(
            dataframe
        )

        print(
            "\nChurn model evaluation"
        )

        print(
            "======================="
        )

        print(
            f"Accuracy : {result.accuracy:.4f}"
        )

        print(
            f"Precision: {result.precision:.4f}"
        )

        print(
            f"Recall   : {result.recall:.4f}"
        )

        print(
            f"F1       : {result.f1:.4f}"
        )

        print(
            f"ROC-AUC  : {result.roc_auc:.4f}"
        )

        print(
            "\nModel saved to:"
        )

        print(
            "data/models/churn_model.joblib"
        )

    finally:

        db.close()


if __name__ == "__main__":
    main()
