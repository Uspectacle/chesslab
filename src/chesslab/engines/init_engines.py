import logging
from typing import Any, Dict, List, Optional

import chess.engine
import numpy as np
import structlog
import torch
from sqlalchemy.orm import Session

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
    player = get_player_by_attributes(
        session=session,
        engine_type="MaiaEngine",
        expected_elo=int(elo),
        options={
            "UCI_Elo": int(elo),
            "OpponentElo": int(opponent_elo) if opponent_elo is not None else int(elo),
            "ModelType": "blitz" if for_blitz else "rapid",
            "Device": "gpu" if use_gpu and torch.cuda.is_available() else "cpu",
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
    players: List[Player],
    aggregator: str = "majority",
    max_concurrent: int = 1,
    create_not_raise: bool = True,
) -> Player:
    player_ids = ",".join([str(player.id) for player in players])

    logger.debug(
        "Getting or creating voting player",
        player_ids=player_ids,
        max_concurrent=max_concurrent,
    )

    expected_elos = [player.expected_elo for player in players]
    mean_elo = sum(expected_elos) / len(expected_elos)

    if aggregator == "top_elo_dictator":
        expected_elo = max(expected_elos)
    elif aggregator == "bottom_elo_dictator":
        expected_elo = min(expected_elos)
    else:
        expected_elo = mean_elo

    player = get_player_by_attributes(
        session=session,
        engine_type="VotingEngine",
        expected_elo=int(expected_elo),
        options={
            "Player_ids": player_ids,
            "Max_concurrent": max_concurrent,
            "Aggregator": aggregator,
        },
        create_not_raise=create_not_raise,
    )
    logger.info(
        "Voting player ready",
        player_id=player.id,
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
        calculated_elo = stockfish_elo(depth)
        logger.debug(
            "No Elo specified, using unlimited strength",
            calculated_elo=calculated_elo,
        )

    player = get_player_by_attributes(
        session=session,
        engine_type="Stockfish",
        expected_elo=int(elo) if elo else stockfish_elo(depth),
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


def get_stockfish_range(
    session: Session,
    min_elo: int = 1320,
    max_elo: int = 2200,
    num_step: int = 3,
) -> list[Player]:
    return [
        get_stockfish_player(session=session, elo=elo)
        for elo in np.linspace(min_elo, max_elo, num_step)
    ]


def get_stockfish_gaussian(
    session: Session,
    mean: float = 1700,
    std_dev: float = 200,
    num_samples: int = 3,
    min_elo: int = 1320,
    max_elo: int = 2200,
    seed: Optional[int] = None,
) -> list[Player]:
    rng = np.random.default_rng(seed) if seed else np.random
    sampled_elos = rng.normal(loc=mean, scale=std_dev, size=num_samples)
    sampled_elos = np.clip(sampled_elos, min_elo, max_elo)
    sampled_elos = sampled_elos.astype(int)
    return [get_stockfish_player(session=session, elo=elo) for elo in sampled_elos]


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
