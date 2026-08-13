from sqlalchemy.orm import Session

from app.db.models import AuditLog


def write_audit_log(
    db: Session,
    user_id: str,
    operation: str,
    tool_name: str,
    requested_message: str,
    success: bool = True,
    error_category: str | None = None,
) -> AuditLog:
    """
    Persist an audit record for an agent operation.

    Audit contract:
        success=True  -> successful operation
        success=False -> failed operation

    If an error category is supplied, the operation is automatically
    treated as unsuccessful. This prevents inconsistent records such as:

        success=True, error_category="llm_error"
    """

    # Defensive consistency rule.
    if error_category is not None:
        success = False

    log = AuditLog(
        user_id=str(user_id or "anonymous"),
        operation=str(operation),
        tool_name=str(tool_name),
        requested_message=str(requested_message),
        success=bool(success),
        error_category=error_category,
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log


def recent_audit_logs(
    db: Session,
    limit: int = 50,
) -> list[AuditLog]:
    """
    Return the most recent audit records.

    The limit is constrained to prevent excessive database reads.
    """

    try:
        requested_limit = int(limit)
    except (TypeError, ValueError):
        requested_limit = 50

    safe_limit = max(
        1,
        min(
            requested_limit,
            500,
        ),
    )

    return (
        db.query(AuditLog)
        .order_by(
            AuditLog.timestamp.desc(),
            AuditLog.id.desc(),
        )
        .limit(safe_limit)
        .all()
    )