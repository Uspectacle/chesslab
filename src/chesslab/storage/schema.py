"""SQLAlchemy models for ChessLab database schema.

Defines the database structure for players, games, moves, evaluations, and batch requests.
Uses PostgreSQL with proper concurrent write support and JSON fields.
"""

from datetime import datetime
from typing import Any, Dict

import structlog
from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from chesslab.storage.db_tools import Base

logger = structlog.get_logger()


class Player(Base):
    """Represents a chess engine or player configuration."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    engine_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    expected_elo: Mapped[int] = mapped_column(nullable=False)
    options: Mapped[Dict[str, Any]] = mapped_column(JSON)
    limit: Mapped[Dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)

    # Games played as white or black
    white_games: Mapped[list["Game"]] = relationship(
        "Game", foreign_keys="Game.white_player_id", back_populates="white_player"
    )
    black_games: Mapped[list["Game"]] = relationship(
        "Game", foreign_keys="Game.black_player_id", back_populates="black_player"
    )

    # Evaluations made by this player
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation", back_populates="evaluator"
    )

    # Requests assigned to this player
    requests: Mapped[list["Request"]] = relationship(
        "Request", back_populates="evaluator"
    )

    def __repr__(self) -> str:
        return f"<Player(id={self.id}, engine_type='{self.engine_type}', expected_elo={self.expected_elo})>"


class Game(Base):
    """Represents a single chess game between two players."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    white_player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    black_player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        default=datetime.now, nullable=False, index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    game_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON)

    white_player: Mapped["Player"] = relationship(
        "Player", foreign_keys=[white_player_id], back_populates="white_games"
    )
    black_player: Mapped["Player"] = relationship(
        "Player", foreign_keys=[black_player_id], back_populates="black_games"
    )
    moves: Mapped[list["Move"]] = relationship(
        "Move", back_populates="game", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_game_players", "white_player_id", "black_player_id"),)

    def __repr__(self) -> str:
        return (
            f"<Game(id={self.id}, white={self.white_player_id}, "
            f"black={self.black_player_id}, result='{self.result}')>"
        )


class Move(Base):
    """Represents a single move in a game."""

    __tablename__ = "moves"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ply_index: Mapped[int] = mapped_column(nullable=False)  # 1..N (half-move number)
    move_number: Mapped[int] = mapped_column(nullable=False)  # Full move number
    is_white: Mapped[bool] = mapped_column(nullable=False)
    fen_before: Mapped[str] = mapped_column(Text, nullable=False)
    uci_move: Mapped[str | None] = mapped_column(
        String(10), nullable=True
    )  # NULL while pending
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)
    played_at: Mapped[datetime | None] = mapped_column(nullable=True)

    game: Mapped["Game"] = relationship("Game", back_populates="moves")
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation", back_populates="move", cascade="all, delete-orphan"
    )
    requests: Mapped[list["Request"]] = relationship(
        "Request", back_populates="move", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_move_game_ply", "game_id", "ply_index"),)

    def __repr__(self) -> str:
        return f"<Move(id={self.id}, game={self.game_id}, ply={self.ply_index}, move='{self.uci_move}')>"


class Evaluation(Base):
    """Represents an engine's evaluation of a specific move/position."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    move_id: Mapped[int] = mapped_column(
        ForeignKey("moves.id"), nullable=False, index=True
    )
    evaluator_id: Mapped[int] = mapped_column(
        ForeignKey("players.id"), nullable=False, index=True
    )
    uci_evaluated: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # Bestmove found
    score: Mapped[float] = mapped_column(Float, nullable=False)
    analysis_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)

    move: Mapped["Move"] = relationship("Move", back_populates="evaluations")
    evaluator: Mapped["Player"] = relationship("Player", back_populates="evaluations")

    __table_args__ = (Index("idx_eval_move_evaluator", "move_id", "evaluator_id"),)

    def __repr__(self) -> str:
        return f"<Evaluation(id={self.id}, move={self.move_id}, evaluator={self.evaluator_id}, score={self.score})>"


class Request(Base):
    """Represents a pending evaluation request for batch processing."""

    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    move_id: Mapped[int] = mapped_column(
        ForeignKey("moves.id"), nullable=False, index=True
    )
    evaluator_id: Mapped[int] = mapped_column(
        ForeignKey("players.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        default=datetime.now, nullable=False, index=True
    )

    # Relationships
    move: Mapped["Move"] = relationship("Move", back_populates="requests")
    evaluator: Mapped["Player"] = relationship("Player", back_populates="requests")

    __table_args__ = (
        Index("idx_request_status_evaluator", "status", "evaluator_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Request(id={self.id}, move_id={self.move_id}, status='{self.status}')>"
        )
