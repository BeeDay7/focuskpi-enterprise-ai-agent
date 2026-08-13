from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.db.database import Base, SessionLocal, engine
from app.db.models import Customer, Sale


SEED = 42

RNG = np.random.default_rng(SEED)

Path("data").mkdir(exist_ok=True)

Base.metadata.create_all(bind=engine)


def generate_customers(count: int = 2000) -> list[Customer]:
    customers: list[Customer] = []

    regions = [
        "North",
        "South",
        "East",
        "West",
        "Central",
    ]

    for index in range(count):

        customer_code = f"CUST-{1001 + index}"

        tenure = int(
            RNG.integers(
                low=3,
                high=73,
            )
        )

        monthly_spend = round(
            float(
                RNG.normal(
                    loc=125,
                    scale=35,
                )
            ),
            2,
        )

        monthly_spend = max(
            40.0,
            monthly_spend,
        )

        support_tickets = int(
            RNG.poisson(
                lam=2.5
            )
        )

        late_payments = int(
            RNG.poisson(
                lam=0.7
            )
        )

        engagement_score = float(
            RNG.uniform(
                0.20,
                0.98,
            )
        )

        # Synthetic churn probability.
        risk_score = (
            -2.2
            - 0.025 * tenure
            + 0.010 * monthly_spend
            + 0.18 * support_tickets
            + 0.40 * late_payments
            - 1.20 * engagement_score
        )

        churn_probability = 1.0 / (
            1.0 + np.exp(-risk_score)
        )

        churned = int(
            RNG.random()
            < churn_probability
        )

        customers.append(
            Customer(
                id=index + 1,
                customer_code=customer_code,
                tenure_months=tenure,
                monthly_spend=monthly_spend,
                support_tickets=support_tickets,
                late_payments=late_payments,
                churned=churned,
            )
        )

    # Guarantee useful class balance.
    churned_count = sum(
        customer.churned
        for customer in customers
    )

    if churned_count < count * 0.15:
        for customer in customers[: int(count * 0.15)]:
            customer.churned = 1

    if churned_count > count * 0.60:
        for customer in customers[int(count * 0.60):]:
            customer.churned = 0

    return customers


def generate_sales(customers: list[Customer]) -> list[Sale]:

    regions = [
        "North",
        "South",
        "East",
        "West",
        "Central",
    ]

    products = [
        "Analytics Platform",
        "Cybersecurity Suite",
        "AI Assistant",
        "Data Integration",
        "Cloud Operations",
        "Automation Pro",
    ]

    months = [
        f"2025-{month:02d}"
        for month in range(1, 13)
    ]

    sales: list[Sale] = []

    sale_id = 1

    for _ in range(5500):

        region = RNG.choice(regions)

        product = RNG.choice(products)

        month = RNG.choice(months)

        amount = float(
            RNG.uniform(
                250,
                18000,
            )
        )

        sales.append(
            Sale(
                id=sale_id,
                region=str(region),
                product=str(product),
                amount=round(
                    amount,
                    2,
                ),
                month=str(month),
            )
        )

        sale_id += 1

    return sales


def main() -> None:

    db: Session = SessionLocal()

    try:

        db.query(Sale).delete()
        db.query(Customer).delete()

        customers = generate_customers()

        sales = generate_sales(
            customers
        )

        db.add_all(customers)
        db.add_all(sales)

        db.commit()

        churned = sum(
            customer.churned
            for customer in customers
        )

        print(
            "Enterprise demo database created."
        )

        print(
            f"Customers: {len(customers):,}"
        )

        print(
            f"Sales records: {len(sales):,}"
        )

        print(
            f"Churned customers: {churned:,}"
        )

        print(
            "Database: data/demo.db"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()