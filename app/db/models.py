from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Sale(Base):
    __tablename__ = "sales"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    region: Mapped[str] = mapped_column(
        String(50),
        index=True,
    )

    product: Mapped[str] = mapped_column(
        String(100),
    )

    amount: Mapped[float] = mapped_column(
        Float,
    )

    month: Mapped[str] = mapped_column(
        String(20),
    )


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    customer_code: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        index=True,
    )

    tenure_months: Mapped[int] = mapped_column(
        Integer,
    )

    monthly_spend: Mapped[float] = mapped_column(
        Float,
    )

    support_tickets: Mapped[int] = mapped_column(
        Integer,
    )

    late_payments: Mapped[int] = mapped_column(
        Integer,
    )

    churned: Mapped[int] = mapped_column(
        Integer,
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        default="anonymous",
    )

    operation: Mapped[str] = mapped_column(
        String(100),
        index=True,
    )

    tool_name: Mapped[str] = mapped_column(
        String(100),
        index=True,
    )

    requested_message: Mapped[str] = mapped_column(
        Text,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )

    success: Mapped[bool] = mapped_column(
        default=True,
    )

    error_category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )