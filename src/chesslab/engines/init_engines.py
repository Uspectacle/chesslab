import logging
from typing import Any, Dict, List, Literal, Optional

import chess.engine
import numpy as np
import structlog
from sqlalchemy.orm import Session

from chesslab.env import get_device
from chesslab.storage import Player, get_session
from chesslab.storage.player_tools import get_player_by_attributes

logger = structlog.get_logger()


def get_random_player(
    session: Session,
    seed: Optional[int] = None,
    create_not_raise: bool = True,
) -> Player:
    logger.debug("Getting or creating random player", seed=seed)

    player = get_player_by_attributes(
        session=session,
        engine_type="RandomEngine",
        expected_elo=300,
        options={"Seed": seed} if seed else {},
        create_not_raise=create_not_raise,
    )
    logger.info(
        "Random player ready",
        player_id=player.id,
        seed=seed,
    )
    return player


def get_maia_player(
    session: Session,
    elo: int,
    opponent_elo: Optional[int] = None,
    for_blitz: bool = False,
    use_gpu: bool = True,
    create_not_raise: bool = True,
) -> Player:
    elo = int(np.clip(elo, 1100, 1900))
    player = get_player_by_attributes(
        session=session,
        engine_type="MaiaEngine",
        expected_elo=elo,
        options={
            "UCI_Elo": elo,
            "OpponentElo": int(opponent_elo) if opponent_elo is not None else elo,
            "ModelType": "blitz" if for_blitz else "rapid",
            "Device": "gpu" if use_gpu and get_device() == "cuda" else "cpu",
        },
        create_not_raise=create_not_raise,
    )

    logger.info(
        "Maia player ready",
        player_id=player.id,
        expected_elo=int(player.expected_elo),
    )

    return player


def get_llm_player(
    session: Session,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    parser: Optional[str] = None,
    quantization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    continuous_conversation: Optional[bool] = None,
    create_not_raise: bool = True,
) -> Player:
    logger.debug("Getting or creating LLM player", model_name=model_name)
    options: dict[str, Any] = {}

    if model_name is not None:
        options["Model"] = model_name

    if system_prompt is not None:
        options["System_Prompt"] = system_prompt

    if user_prompt is not None:
        options["User_Prompt"] = user_prompt

    if parser is not None:
        options["Parser"] = parser

    if quantization is not None:
        options["Quantization"] = quantization

    if max_tokens is not None:
        options["Max_Tokens"] = max_tokens

    if temperature is not None:
        options["Temperature"] = temperature

    if max_retries is not None:
        options["Max_Retries"] = max_retries

    if continuous_conversation is not None:
        options["Continuous_Conversation"] = continuous_conversation

    player = get_player_by_attributes(
        session=session,
        engine_type="LlmEngine",
        expected_elo=300,
        options=options,
        create_not_raise=create_not_raise,
    )
    logger.info("LLM player ready", player_id=player.id)

    return player


def get_voting_player(
    session: Session,
    players: Optional[List[Player]] = None,
    *,
    aggregator: str = "majority",
    weights: Optional[List[float] | Literal["elo"]] = None,
    max_concurrent: int = 1,
    crowd_kind: Literal[
        "Stockfish gaussian", "MadChess gaussian", "Maia gaussian", "Explicit"
    ] = "Explicit",
    crowd_size: int = 10,
    crowd_mean_elo: int = 1500,
    crowd_std_dev: int = 200,
    seed: Optional[int] = None,
    create_not_raise: bool = True,
    expected_elo: Optional[int] = None,
) -> Player:
    """
    Get or create a VotingEngine Player.

    Two modes:

    1) EXPLICIT CROWD (default)
       - Provide `players=[...]`
       - Stored via option: Crowd_ids="1,2,3"

    2) GAUSSIAN-GENERATED CROWD
       - Set players=None
       - Must provide: crowd_kind + crowd_size
       - Optionally: crowd_mean_elo, crowd_std_dev, seed
    """

    # ---------- Validate mode ----------
    if players is not None and len(players):
        # Explicit players mode
        player_ids = ",".join(str(p.id) for p in players)
        crowd_ids_option = player_ids
    else:
        # Gaussian mode validation
        if crowd_kind is None:
            raise ValueError(
                "crowd_kind must be provided when players is None (Gaussian mode)"
            )
        if crowd_size is None:
            raise ValueError(
                "crowd_size must be provided when players is None (Gaussian mode)"
            )

        player_ids = "GAUSSIAN"
        crowd_ids_option = "None"  # Important: triggers Gaussian generation

    logger.debug(
        "Getting or creating voting player",
        player_ids=player_ids,
        max_concurrent=max_concurrent,
        gaussian=players is not None and len(players),
        crowd_kind=crowd_kind,
    )

    # ---------- Expected ELO ----------
    if expected_elo is None:
        if players is not None and len(players):
            expected_elos = [p.expected_elo for p in players]
            expected_elo = int(sum(expected_elos) / len(expected_elos))
        else:
            # In Gaussian mode, default to mean if provided, else 1500
            expected_elo = crowd_mean_elo

    # ---------- Build options dict ----------
    options = {
        # Common options
        "Max_concurrent": max_concurrent,
        "Aggregator": aggregator,
        "Weights": weights if weights else "None",
    }

    # ---------- Add Gaussian options if in that mode ----------
    if players is not None and len(players):
        # Crowd selection
        options.update({"Crowd_ids": crowd_ids_option})

    if players is not None and len(players):
        options.update(
            {
                "Crowd_kind": crowd_kind,
                "Crowd_size": crowd_size,
                "Crowd_mean_elo": crowd_mean_elo,
                "Crowd_std_dev": crowd_std_dev,
                "Seed": seed or 0,
            }
        )

    player = get_player_by_attributes(
        session=session,
        engine_type="VotingEngine",
        expected_elo=expected_elo,
        options=options,
        create_not_raise=create_not_raise,
    )

    logger.info(
        "Voting player ready",
        player_id=player.id,
        mode="gaussian" if players is not None and len(players) else "explicit",
        player_ids=player_ids,
        max_concurrent=max_concurrent,
    )

    return player


def stockfish_elo(depth: int) -> int:
    elo = 66 * depth + 1570

    return int(round(elo))


def get_stockfish_player(
    session: Session,
    elo: Optional[int | float] = None,
    depth: int = 10,
    create_not_raise: bool = True,
) -> Player:
    calculated_elo = stockfish_elo(depth)
    if elo:
        elo = int(np.clip(elo, 1320, calculated_elo))
        options: Dict[str, Any] = {
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
        }
        logger.debug("Using Elo limit strength", elo=elo)
    else:
        elo = calculated_elo
        options: Dict[str, Any] = {
            "UCI_LimitStrength": False,
        }
        logger.debug(
            "No Elo specified, using unlimited strength",
            calculated_elo=calculated_elo,
        )

    player = get_player_by_attributes(
        session=session,
        engine_type="Stockfish",
        expected_elo=elo,
        options=options,
        limit=chess.engine.Limit(depth=depth),
        create_not_raise=create_not_raise,
    )

    logger.info(
        "Stockfish player ready",
        player_id=player.id,
        expected_elo=player.expected_elo,
    )

    return player


def get_arasan_player(
    session: Session,
    elo: Optional[int | float] = None,
    create_not_raise: bool = True,
) -> Player:
    calculated_elo = 3450
    if elo:
        elo = int(np.clip(elo, 1320, calculated_elo))
        options: Dict[str, Any] = {
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
        }
        logger.debug("Using Elo limit strength", elo=elo)
    else:
        elo = calculated_elo
        options: Dict[str, Any] = {
            "UCI_LimitStrength": False,
        }
        logger.debug(
            "No Elo specified, using unlimited strength",
            calculated_elo=calculated_elo,
        )

    player = get_player_by_attributes(
        session=session,
        engine_type="Arasan",
        expected_elo=elo,
        options=options,
        create_not_raise=create_not_raise,
    )

    logger.info(
        "Arasan player ready",
        player_id=player.id,
        expected_elo=player.expected_elo,
    )

    return player


def get_madchess_player(
    session: Session,
    elo: Optional[int | float] = None,
    time: float = 0.3,
    create_not_raise: bool = True,
) -> Player:
    if bool(elo):
        options: Dict[str, Any] = {
            "UCI_LimitStrength": True,
            "UCI_Elo": int(elo),
        }
        logger.debug("Using Elo limit strength", elo=elo)
    else:
        options: Dict[str, Any] = {
            "UCI_LimitStrength": False,
        }

    player = get_player_by_attributes(
        session=session,
        engine_type="MadChess",
        expected_elo=int(elo) if elo else 2800,
        options=options,
        limit=chess.engine.Limit(time=time),
        create_not_raise=create_not_raise,
    )

    logger.info(
        "MadChess player ready",
        player_id=player.id,
        expected_elo=player.expected_elo,
    )

    return player


def get_distilled_stockfish(session: Session, ratio: float, depth: int = 10) -> Player:
    return get_voting_player(
        session=session,
        players=[
            get_random_player(session=session),
            get_stockfish_player(session=session, depth=depth),
        ],
        aggregator="randomized",
        weights=[1 - ratio, ratio],
        expected_elo=int(ratio * (stockfish_elo(depth) - 300) + 300),
    )


def get_distilled_maia(session: Session, ratio: float, elo: int = 1100) -> Player:
    return get_voting_player(
        session=session,
        players=[
            get_random_player(session=session),
            get_maia_player(session=session, elo=elo),
        ],
        aggregator="randomized",
        weights=[1 - ratio, ratio],
        expected_elo=int(ratio * (elo - 300) + 300),
    )


def get_stockfish_range(
    session: Session,
    min_elo: int = 1320,
    max_elo: int = 2200,
    num_step: int = 10,
) -> list[Player]:
    return [
        get_stockfish_player(session=session, elo=elo)
        for elo in np.linspace(min_elo, max_elo, num_step)
    ]


def get_madchess_range(
    session: Session,
    min_elo: int = 600,
    max_elo: int = 2600,
    num_step: int = 11,
) -> list[Player]:
    return [
        get_madchess_player(session=session, elo=elo)
        for elo in np.linspace(min_elo, max_elo, num_step)
    ]


def get_maia_range(
    session: Session,
    min_elo: int = 1100,
    max_elo: int = 1900,
    num_step: int = 9,
) -> list[Player]:
    return [
        get_maia_player(session=session, elo=elo)
        for elo in np.linspace(min_elo, max_elo, num_step)
    ]


if __name__ == "__main__":
    logger.info("Initializing engines script")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    with get_session() as session:
        logger.info("Database session created")

        logger.info("Creating Stockfish player with Elo 1320")
        white_player = get_stockfish_player(
            session=session,
            elo=1320,
        )
        logger.info("White player created", player_id=white_player.id)

        logger.info("Creating random player with seed 2")
        black_player = get_random_player(
            session=session,
            seed=2,
        )
        logger.info("Black player created", player_id=black_player.id)

        print(white_player)
        print(black_player)
