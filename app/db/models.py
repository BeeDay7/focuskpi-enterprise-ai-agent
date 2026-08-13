from sqlalchemy import String, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base

class Sale(Base):
    __tablename__ = "sales"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region: Mapped[str] = mapped_column(String(50), index=True)
    product: Mapped[str] = mapped_column(String(100))
    amount: Mapped[float] = mapped_column(Float)
    month: Mapped[str] = mapped_column(String(20))

class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_code: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    tenure_months: Mapped[int] = mapped_column(Integer)
    monthly_spend: Mapped[float] = mapped_column(Float)
    support_tickets: Mapped[int] = mapped_column(Integer)
    late_payments: Mapped[int] = mapped_column(Integer)
    churned: Mapped[int] = mapped_column(Integer)
