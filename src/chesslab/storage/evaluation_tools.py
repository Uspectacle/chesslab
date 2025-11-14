"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.orm import Session

from chesslab.storage.schema import Evaluation

logger = structlog.get_logger()


def create_evaluation(
    session: Session,
    move_id: int,
    evaluator_id: int,
    uci_evaluated: str,
    score: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Evaluation:
    """Create a new evaluation record.

    Args:
        session: Database session
        move_id: Move ID
        evaluator_id: Evaluator player ID
        uci_evaluated: Best move found by evaluator
        score: Evaluation score
        metadata: Optional evaluation metadata

    Returns:
        Created evaluation instance
    """
    evaluation = Evaluation(
        move_id=move_id,
        evaluator_id=evaluator_id,
        uci_evaluated=uci_evaluated,
        score=score,
        metadata=metadata or {},
    )
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)

    return evaluation


def get_move_evaluations(session: Session, move_id: int) -> List[Evaluation]:
    """Get all evaluations for a specific move.

    Args:
        session: Database session
        move_id: Move ID

    Returns:
        List of evaluation instances
    """
    return session.query(Evaluation).filter(Evaluation.move_id == move_id).all()
