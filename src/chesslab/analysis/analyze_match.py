"""Database-driven analysis tools for ChessLab.

Retrieves game data from the database and performs statistical analysis,
ELO estimation, and visualization without running new games.
"""

from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from chesslab.analysis.elo_tools import elo_from_mean_score, expected_score
from chesslab.arena.init_engines import get_or_create_random_player
from chesslab.storage import (
    Player,
    get_head_to_head_games,
    get_session,
)


class MatchAnalysis:
    """Analysis of games between two specific players."""

    def __init__(self, session: Session, player: Player, opponent: Player):
        self.session = session
        self.player = player
        self.opponent = opponent

        # Get results

        games = get_head_to_head_games(
            session=session, player1_id=player.id, player2_id=opponent.id
        )

        self.games = [game for game in games if game.result]

        player_score = 0.0
        opponent_score = 0.0

        for game in self.games:
            if game.result == "1-0":
                if game.white_player_id == player.id:
                    player_score += 1.0
                else:
                    opponent_score += 1.0
            elif game.result == "0-1":
                if game.black_player_id == player.id:
                    player_score += 1.0
                else:
                    opponent_score += 1.0
            elif game.result == "1/2-1/2":
                player_score += 0.5
                opponent_score += 0.5

        return player_score, opponent_score, len(games)

    @property
    def player_score(self) -> float:
        player_score = 0.0

        for game in self.games:
            if game.result == "1-0" and game.white_player_id == self.player.id:
                player_score += 1.0
            elif game.result == "0-1" and game.black_player_id == self.player.id:
                player_score += 1.0
            else:
                player_score += 0.5

        return player_score

    @property
    def opponent_score(self) -> float:
        return self.num_games - self.player_score

    @property
    def num_games(self) -> int:
        return len(self.games)

    @property
    def player_mean_score(self) -> float:
        """Average score of the player."""
        if self.num_games == 0:
            return 0.0
        return self.player_score / self.num_games

    @property
    def opponent_mean_score(self) -> float:
        """Average score of the opponent."""
        if self.num_games == 0:
            return 0.0
        return self.opponent_score / self.num_games

    @property
    def expected_player_score(self) -> float:
        """Expected score based on declared ELOs."""
        return expected_score(self.player.elo, self.opponent.elo)

    @property
    def estimated_player_elo(self) -> float:
        """Estimated ELO based on actual results."""
        if self.num_games == 0:
            return self.player.elo
        return elo_from_mean_score(self.player_mean_score, self.opponent.elo)

    @property
    def report(self) -> str:
        """Generate text report."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(
            f"Match Analysis: Player {self.player.id} vs Player {self.opponent.id}"
        )
        lines.append("=" * 60)
        lines.append(f"Player: {self.player.engine_type} (ID: {self.player.id})")
        lines.append(f"Opponent: {self.opponent.engine_type} (ID: {self.opponent.id})")
        lines.append(f"Games played: {self.num_games}")
        lines.append("")
        lines.append(f"Player Score: {self.player_score:.1f}")
        lines.append(f"Opponent Score: {self.opponent_score:.1f}")
        lines.append(f"Player Mean Score: {self.player_mean_score:.3f}")
        lines.append("")
        lines.append(f"Player Declared ELO: {int(self.player.elo)}")
        lines.append(f"Opponent Declared ELO: {int(self.opponent.elo)}")
        lines.append(f"Expected Player Score: {self.expected_player_score:.3f}")
        lines.append(f"Estimated Player ELO: {int(self.estimated_player_elo)}")
        lines.append(
            f"ELO Difference: {int(self.estimated_player_elo - self.player.elo):+d}"
        )
        lines.append("")

        return "\n".join(lines)


def analyze_match(
    session: Session,
    player: Player,
    opponent: Player,
    output_dir: Optional[str] = None,
) -> MatchAnalysis:
    try:
        analysis = MatchAnalysis(session=session, player=player, opponent=opponent)

        print(analysis.report)

        if output_dir:
            output_path = Path(output_dir) / f"match_{player.id}_vs_{opponent.id}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(analysis.report)

        return analysis

    finally:
        session.close()


if __name__ == "__main__":
    with get_session() as session:
        player = get_or_create_random_player(
            session=session,
            seed=1,
        )
        opponent = get_or_create_random_player(
            session=session,
            seed=2,
        )

        analyze_match(
            session=session,
            player=player,
            opponent=opponent,
            output_dir="analysis_results",
        )
