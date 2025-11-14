"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from typing import List

import structlog
from sqlalchemy.orm import Session

from chesslab.storage.schema import Request

logger = structlog.get_logger()


def create_request(session: Session, move_id: int, evaluator_id: int) -> Request:
    """Create a new evaluation request.

    Args:
        session: Database session
        move_id: Move ID to evaluate
        evaluator_id: Evaluator player ID

    Returns:
        Created request instance
    """
    request = Request(move_id=move_id, evaluator_id=evaluator_id, status="pending")
    session.add(request)
    session.commit()
    session.refresh(request)

    return request


def get_pending_requests(
    session: Session, evaluator_id: int, limit: int = 20
) -> List[Request]:
    """Get pending evaluation requests for a specific evaluator.

    Args:
        session: Database session
        evaluator_id: Evaluator player ID
        limit: Maximum number of requests

    Returns:
        List of pending request instances
    """
    return (
        session.query(Request)
        .filter(Request.evaluator_id == evaluator_id, Request.status == "pending")
        .order_by(Request.created_at)
        .limit(limit)
        .all()
    )


def mark_requests_processing(session: Session, request_ids: List[int]) -> int:
    """Mark requests as processing to prevent race conditions.

    Args:
        session: Database session
        request_ids: List of request IDs to mark

    Returns:
        Number of requests updated
    """
    count = (
        session.query(Request)
        .filter(Request.id.in_(request_ids), Request.status == "pending")
        .update({Request.status: "processing"}, synchronize_session=False)
    )
    session.commit()

    logger.info("Requests marked processing", count=count)
    return count


def mark_requests_completed(session: Session, request_ids: List[int]) -> int:
    """Mark requests as completed.

    Args:
        session: Database session
        request_ids: List of request IDs to mark

    Returns:
        Number of requests updated
    """
    count = (
        session.query(Request)
        .filter(Request.id.in_(request_ids))
        .update({Request.status: "completed"}, synchronize_session=False)
    )
    session.commit()

    logger.info("Requests marked completed", count=count)
    return count
