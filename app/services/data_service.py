from sqlalchemy import select, func
from sqlalchemy.orm import Session
from app.db.models import Sale, Customer

def sales_by_region(db: Session) -> list[dict]:
    rows = db.execute(select(Sale.region, func.sum(Sale.amount).label("total")).group_by(Sale.region).order_by(func.sum(Sale.amount).desc())).all()
    return [{"region": r, "total_sales": round(float(t), 2)} for r, t in rows]

def customer_by_code(db: Session, code: str):
    return db.execute(select(Customer).where(Customer.customer_code == code)).scalar_one_or_none()

def all_customers(db: Session):
    return db.execute(select(Customer)).scalars().all()
