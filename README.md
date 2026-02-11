# ChessLab

A modular, open-source chess engine testing and evaluation framework with PostgreSQL storage, parallel game execution, and comprehensive statistical analysis tools.

## Features

- **Multiple Engine Types**: Stockfish, Arasan, MadChess, Maia2, LLM, voting ensembles
- **Parallel Game Execution**: Run hundreds of games concurrently
- **Persistent Storage**: Game history, moves, evaluations, engine configurations
- **Statistical Analysis**: Elo estimation, confidence intervals, R² analysis, effect size measures
- **LLM Integration**: HuggingFace models with customizable prompts and template variables
- **Ensemble Methods**: Combine multiple engines with diverse voting strategies
- **PGN Support**: Import/export games in standard PGN format

## Acknowledgements

- LLMs were used during the development of this project
- **Maia2**: Neural network model for human-like chess play - [CSSLab/maia2](https://github.com/CSSLab/maia2)
- **Stockfish**: Classical UCI chess engine - [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish) (included as a git submodule)
- **Arasan**: Classical UCI chess engine - [jdart1/arasan-chess](https://github.com/jdart1/arasan-chess) (included as a git submodule)
- **MadChess**: Classical UCI chess engine - [ekmadsen/MadChess](https://github.com/ekmadsen/MadChess) (included as a git submodule)
- **Chess.py**: Pure Python chess library - [niklasf/python-chess](https://github.com/niklasf/python-chess)

## Quick Start

### 1. Prerequisites

- **Python 3.13+**
- **PostgreSQL 14+** (running locally)
- **uv** package manager _(recommended)_

> **Note for Linux users:** If you just installed PostgreSQL, ensure it is initialized and running:
>
> ```bash
> sudo postgresql-setup --initdb
> sudo systemctl enable --now postgresql
> ```

### 2. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/Uspectacle/chesslab.git
cd chesslab
```

_If already cloned without submodules:_ `git submodule update --init --recursive`

### 3. Install Python Dependencies

```bash
# Recommended
uv sync

# Or with pip
pip install -e .
```

### 4. Build Engines

Stockfish, Arasan and MadChess are included as a submodule and must be compiled for your specific architecture. (Maia is handled via Python dependencies).

```bash
cd src/third_party/stockfish/src/
make -j profile-build
cd -
```

```bash
cd src/third_party/arasan/src
git submodule update --init --recursive
make CC=clang BUILD_TYPE=avx2-bmi2 profiled
cd -
```

```bash
cd src/third_party/MadChess/
dotnet publish src/Engine/Engine.csproj -c Release -r linux-x64 --self-contained true /p:PublishSingleFile=true /p:UseAppHost=true
cd -
```

### 5. Configure Environment

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://chesslab:chesslab_dev@localhost:5432/chesslab
```

### 6. Create and Configure the Database

Run the following block as the `postgres` superuser to create the user, the database, and set the correct schema permissions (required for PostgreSQL 15+).

```bash
sudo -u postgres psql <<EOF
CREATE USER chesslab WITH PASSWORD 'chesslab_dev';
CREATE DATABASE chesslab OWNER chesslab;
GRANT ALL PRIVILEGES ON DATABASE chesslab TO chesslab;
\c chesslab
GRANT ALL ON SCHEMA public TO chesslab;
EOF
```

### 7. Initialize the Database Schema

This command sets up the tables and structures required for the experiments.

```bash
uv run -m chesslab.storage.setup_db
```

### 8. Access the Database Shell (Optional)

To check your data manually, connect using the project user:

```bash
psql -U chesslab -d chesslab -h localhost
```

### 9. Run Experiments

You are now ready to run the experiment suite:

```bash
# Baseline Stockfish Elo coherence testing
uv run -m chesslab.experiment.coherence_stockfish

# MadChess Elo coherence validation
uv run -m chesslab.experiment.coherence_madchess

# Maia engine Elo coherence validation
uv run -m chesslab.experiment.coherence_maia

# Stockfish against MadChess
uv run -m chesslab.experiment.stockfish_on_madchess

# Maia against MadChess
uv run -m chesslab.experiment.maia_on_madchess

# Voting ensemble testing with MadChess crowds
uv run -m chesslab.experiment.voting_madchess

# LLM prompt optimization
uv run -m chesslab.experiment.llm_prompt
```

## Usage Examples

### Run a simple game

```python
import asyncio

from chesslab.analysis.analyze_game import GameAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.arena.run_game import create_game, run_game
from chesslab.engines.init_engines import get_random_player, get_stockfish_player, get_session

with get_session() as session:
    # Create players
    stockfish = get_stockfish_player(session=session, elo=1500)
    random = get_random_player(session=session)

    # Initialize a game
    game = create_game(
        session=session,
        white_player_id=stockfish.id,
        black_player_id=random.id
    )

    # Run the game
    asyncio.run(run_game(session=session, game=game))

    # Analyze the game
    with Evaluator() as evaluator:
        analysis = GameAnalysis(evaluator=evaluator, game=game)
        print(analysis.report)
```

### Run a match between two engines

```python
import asyncio
from chesslab.analysis.analyze_match import MatchAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.arena.run_match import get_or_create_match, run_multiple_games
from chesslab.engines.init_engines import get_stockfish_player, get_maia_player, get_session

with get_session() as session:
    white = get_stockfish_player(session=session, elo=1600)
    black = get_maia_player(session=session, elo=1400)

    # Create 100 games between the two players
    games = get_or_create_match(
        session=session,
        white_player_id=white.id,
        black_player_id=black.id,
        num_games=100,
        alternate_color=True
    )

    # Run all games in parallel (max 8 concurrent)
    asyncio.run(run_multiple_games(session=session, games=games, max_concurrent=8))

    # Analyze results with statistical tests
    with Evaluator() as evaluator:
        analysis = MatchAnalysis(
            session=session,
            evaluator=evaluator,
            player_1=white,
            player_2=black,
            num_games=100
        )
        print(analysis.report)
```

### Evaluate a Player Across a Range of Opponents

```python
from chesslab.analysis.analyze_range import RangeAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.arena.run_match import run_range
from chesslab.engines.init_engines import (
    get_llm_player,
    get_madchess_range,
    get_session
)

with get_session() as session:
    # Create LLM player
    llm_player = get_llm_player(
        session=session,
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    )

    # Create range of MadChess opponents (600-2600 Elo, 3 steps)
    opponents = get_madchess_range(
        session=session,
        min_elo=600,
        max_elo=2600,
        num_step=3
    )

    # Run all matches
    run_range(
        session=session,
        players=[llm_player],
        opponents=opponents,
        num_games=20,
        max_concurrent=4
    )

    # Analyze performance across the range
    with Evaluator() as evaluator:
        analysis = RangeAnalysis(
            session=session,
            evaluator=evaluator,
            player=llm_player,
            opponents=opponents,
            num_games=20
        )
        print(analysis.report)
        analysis.plot_scores("output_dir")
```

## Project Architecture

### Engine Types and Implementations

- **Stockfish** (`third_party/stockfish`): Classical UCI protocol engine ([official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish))
- **Arasan** (`third_party/arasan`): Classical UCI protocol engine ([jdart1/arasan-chess](https://github.com/jdart1/arasan-chess))
- **MadChess** (`third_party/MadChess`): Classical UCI protocol engine ([ekmadsen/MadChess](https://github.com/ekmadsen/MadChess))
- **Random Engine** (`engines/random_engine.py`): Baseline random legal move generator
- **Maia2 Engine** (`engines/maia_engine.py`): Neural network-based engine trained to simulate human play across skill levels ([CSSLab/maia2](https://github.com/CSSLab/maia2))
- **LLM Engine** (`engines/llm_engine.py`): Large language model with customizable prompts
- **Voting Engine** (`engines/voting_engine.py`): Ensemble combinator aggregating decisions from multiple engines via voting strategies

### Core Module Structure

#### **storage/** - Data Persistence Layer

Handles all database interactions and schema management:

- `schema.py`: SQLAlchemy ORM models (Player, Game, Move, Evaluation, Request)
- `db_tools.py`: Session management and connection pooling
- `game_tools.py`: Game record CRUD operations
- `move_tools.py`: Move sequence and board state management
- `pgn_tools.py`: PGN parsing, import, and export utilities
- `player_tools.py`: Engine/player configuration persistence
- `evaluation_tools.py`: Engine evaluation result storage
- `setup_db.py`: Schema initialization and migration

#### **engines/** - Engine Implementations

UCI-compatible and custom engine interfaces:

- `base_engine.py`: Abstract base class defining engine interface
- `init_engines.py`: Factory functions for creating and retrieving engine instances
- `compile_engines.py`: PyInstaller tools for creating standalone executables
- `storage_tools.py`: Protocol management and serialization
- **options/**: UCI option configuration system
  - `options.py`: Base Option classes (Spin, Combo, Button, String)
  - `prompts.py`: System/user prompt templates for LLM engines
  - `prompt_variables.py`: Template variable substitution (Elo, FEN, legal moves, PGN, etc.)
  - `parsers.py`: Move parsing strategies for LLM outputs
  - `aggregators.py`: Voting and ensemble aggregation methods

#### **arena/** - Game Execution Engine

Orchestrates single games and tournaments:

- `run_game.py`: Asynchronous single game execution with UCI protocol communication
- `run_match.py`: Tournament coordination, parallel game scheduling, and match generation
- Supports concurrent game execution with configurable concurrency limits

#### **analysis/** - Statistical Analysis Tools

Comprehensive post-game evaluation:

- `analyze_game.py`: Single game analysis and move evaluation
- `analyze_match.py`: Head-to-head statistics, Elo estimation, hypothesis testing (t-tests, z-tests)
- `analyze_range.py`: Performance across multiple opponents with weighted analysis
- `evaluator.py`: Engine evaluation wrapper (Stockfish engine for position analysis)
- `elo_tools.py`: Elo calculations, expected scores, parameter estimation
- `stat_tools.py`: Statistical utilities (significance testing, confidence intervals)

#### **experiment/** - Reproducible Research Scripts

Pre-built experimental pipelines:

- `verify_stockfish.py`: Baseline validation of Stockfish across Elo range
- `verify_maia.py`: Maia2 engine evaluation and skill coverage
- `llm_prompt.py`: LLM prompt optimization and model comparison
- `majority_voting.py`: Ensemble voting strategy analysis

### Database Schema

```tree
players (engine configurations)
├── player_options (UCI options, prompts, hyperparameters)
└── games (match results, metadata)
    ├── moves (individual moves in game sequence)
    │   └── evaluations (engine analysis of positions)
    └── requests (batch LLM evaluation queue for async processing)
```

**Key Tables:**

- `players`: Engine type, name, Elo, creation timestamp
- `player_options`: Option name, value, engine-specific settings
- `games`: White/Black player IDs, result, PGN, metadata
- `moves`: Sequence, SAN, UCI, board state, move quality
- `evaluations`: Engine evaluation score, depth, best continuation
- `requests`: Batch processing queue for LLM inference

## Citation

If you use ChessLab in research, please cite as:

```bibtex
@software{chesslab2024,
  title={ChessLab: A Modular Chess Engine Testing and Evaluation Framework},
  author={Larsen, Mario},
  year={2024},
  url={https://github.com/Uspectacle/chesslab}
}
```

## License and Contributing

This project is under GNU v3 License (see [LICENSE.txt](LICENSE.txt)).

Contributions are welcome!
Please submit feature requests, suggestions and report issues on [GitHub Issues](https://github.com/Uspectacle/chesslab/issues)
