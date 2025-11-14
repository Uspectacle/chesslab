# ChessLab

A modular chess engine testing framework with PostgreSQL storage, parallel game execution, and comprehensive analysis tools.

## Features

- **Multiple engine types**: Random, Stockfish wrappers, batch evaluators, alternating engines
- **Parallel game execution**: Run hundreds of games concurrently
- **PostgreSQL storage**: Persistent game history, moves, and evaluations
- **PGN import/export**: Standard format support
- **Analysis tools**: Elo estimation, statistics, visualization

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Setup PostgreSQL database:

```bash
createdb chesslab
python -m scripts.setup_db
```

3. Compile engines to .exe (optional):

```bash
python -m scripts.compile_engines
```

## Usage

### Run a tournament

```python
from arena.runner import TournamentRunner
from engines.random_engine import RandomEngine
from engines.uci_wrapper import StockfishEngine

runner = TournamentRunner()
runner.add_player(RandomEngine())
runner.add_player(StockfishEngine(skill_level=1))
runner.run_tournament(games_per_pairing=100, concurrency=10)
```

### Analyze results

```python
from analysis.example import plot_player_scores

plot_player_scores(player_id_1=1, player_id_2=2)
```

## Architecture

- **engines/**: Engine implementations and wrappers
- **arena/**: Tournament coordination and parallel execution
- **storage/**: Database models, migrations, and PGN tools
- **analysis/**: Statistics, Elo calculation, and visualization

## Quick Start

### 1. Install dependencies

```bash
poetry install
```

### 2. Setup database

```bash
createdb chesslab
python -m scripts.setup_db
```

### 3. Run demo

```bash
python demo.py
```

### 4. Export results

```bash
python -m storage.pgn_tools export games.pgn --limit 100
```

## Usage Examples

### Run a simple match

```python
from engines.random_engine import RandomEngine
from engines.uci_wrapper import StockfishWrapper
from arena.runner import TournamentRunner

runner = TournamentRunner()

# Register players
sf_id = runner.register_player("stockfish", {"skill_level": 5})
random_id = runner.register_player("random", {})

# Create engines
stockfish = StockfishWrapper(skill_level=5)
random = RandomEngine()

# Run 100 games with 10 concurrent
results = runner.run_match(
    stockfish, random,
    sf_id, random_id,
    games=100,
    alternate_colors=True
)
```

### Analyze results

```python
from storage.models import create_db_engine, create_session
from storage.access import get_player_statistics
from analysis.example import plot_player_scores, estimate_elo_ratings

engine = create_db_engine()
session = create_session(engine)

# Get statistics
stats = get_player_statistics(session, player_id=1)
print(f"Win rate: {stats['wins'] / stats['total_games'] * 100:.1f}%")

# Plot head-to-head
plot_player_scores(player_id_1=1, player_id_2=2)

# Estimate Elo ratings
elos = estimate_elo_ratings(session, [1, 2, 3, 4])
```

### Create custom engines

```python
from engines.alternator import AlternatorEngine
from engines.random_engine import RandomEngine
from engines.uci_wrapper import StockfishWrapper

# Mix Stockfish and random moves
alternator = AlternatorEngine(
    engines=[
        StockfishWrapper(skill_level=10),
        RandomEngine()
    ],
    mode="sequential"  # Alternates each move
)
```

## Milestones

### Completed ✓

1. ✓ Project scaffold + DB schema
2. ✓ PGN import/export
3. ✓ Basic arena runner
4. ✓ Parallel execution (10-20 concurrent games)
5. ✓ Random engine + UCI protocol
6. ✓ Worstfish (inverted Stockfish)
7. ✓ Alternating engine (mixes strategies)
8. ✓ Basic analysis tools (Elo, statistics, plots)

### Planned ⏳

9. ⏳ Batch evaluation engine (wait for N requests, evaluate once)
10. ⏳ Gemini API integration (batch position evaluation)
11. ⏳ Engine compilation with PyInstaller
12. ⏳ Web dashboard for live game monitoring

## Project Structure Details

### engines/

- `base.py`: Engine protocol interface
- `random_engine.py`: Random move generator
- `worstfish.py`: Inverted Stockfish (plays worst moves)
- `alternator.py`: Mixes multiple engines
- `uci_wrapper.py`: Generic UCI engine wrapper
- `batch.py`: Batch evaluation (planned)
- `gemini.py`: Gemini API engine (planned)

### arena/

- `runner.py`: Tournament coordinator with parallel execution

### storage/

- `models.py`: SQLAlchemy database models
- `schema.sql`: Reference SQL schema
- `access.py`: High-level database operations
- `pgn_tools.py`: Import/export PGN files

### analysis/

- `example.py`: Statistics, Elo calculation, visualization

### scripts/

- `setup_db.py`: Initialize database schema
- `compile_engines.py`: Build standalone executables

## Technical Notes

- **Concurrency**: 10-20 games recommended to avoid resource exhaustion
- **Database**: PostgreSQL with proper concurrent write support
- **Engine pooling**: Future optimization for better resource management
- **UCI protocol**: All engines compatible with standard chess GUIs
- **Async support**: Consider adding async/await for better performance (future)

## Database Schema

```
players → games ← moves ← evaluations
                    ↓
                 requests (batch processing)
```

- **players**: Engine configurations
- **games**: Match results and metadata
- **moves**: Individual moves with FEN positions
- **evaluations**: Engine analysis of positions
- **requests**: Batch evaluation queue with status tracking

# ChessLab Quick Start

Get ChessLab running in 5 minutes with zero PostgreSQL knowledge required!

## Prerequisites

Just need these two things:
1. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
2. **Python 3.10+** - [Download here](https://www.python.org/downloads/)

## Installation

### 1. Clone and setup
```bash
git clone <your-repo-url>
cd chesslab

# Install Python dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
```

### 2. Start the database
```bash
make db-up
# Wait a few seconds for database to initialize...
```

### 3. Initialize the database schema
```bash
make setup
```

That's it! You're ready to go 🎉

## Running Your First Tournament

```python
from chesslab.tournament import TournamentRunner

# Create tournament runner (uses Docker database automatically)
tournament = TournamentRunner()

# Register two players
stockfish_id = tournament.register_player("stockfish", {"Depth": 10})
random_id = tournament.register_player("RandomEngine")

# Run a match - 10 games with alternating colors
results = tournament.run_match(
    white_player_id=stockfish_id,
    black_player_id=random_id,
    games=10,
    alternate_colors=True
)

# View results
for result in results:
    print(f"Game {result.game_id}: {result.result}")
```

## Useful Commands

```bash
# Database management
make db-up       # Start database
make db-down     # Stop database
make db-reset    # Clear all data and restart
make db-shell    # Open PostgreSQL terminal
make db-logs     # View database logs

# Optional: Start with web UI
make pgadmin     # Access at http://localhost:5050
```

## What Just Happened?

When you ran `make db-up`, Docker:
1. Downloaded PostgreSQL (only happens once)
2. Started it in a container
3. Created the `chesslab` database
4. Made it available at `localhost:5432`

All your data is safely stored in a Docker volume, so it persists even if you stop the container.

## Troubleshooting

### "Port 5432 already in use"
You have PostgreSQL already running locally. Either:
- Stop it: `sudo service postgresql stop` (Linux) or Services app (Windows)
- Or change port in `docker-compose.yml` to `5433:5432`

### Docker Desktop not starting
- Make sure WSL2 is enabled (Windows)
- Restart Docker Desktop
- Check Docker is running: `docker ps`

### Can't connect to database
```bash
# Check database is running
docker ps

# View logs for errors
make db-logs

# Restart fresh
make db-reset
```

## Where's My Data?

All database data is stored in a Docker volume called `chesslab_postgres_data`.

To see it: `docker volume ls`

To backup: `docker run --rm -v chesslab_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/db-backup.tar.gz /data`

To completely delete: `docker compose down -v` (⚠️ destroys all data!)

## Next Steps

- Check out `examples/` for more usage patterns
- Read `DATABASE_SETUP.md` for advanced configuration
- View games in pgAdmin: `make pgadmin`

## Getting Help

If something isn't working:
1. Check `make db-logs` for errors
2. Try `make db-reset` to start fresh
3. Open an issue on GitHub
# ChessLab Database Setup

This project uses PostgreSQL in a Docker container - no local PostgreSQL installation needed!

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Python 3.10+ with dependencies installed

## Quick Start

### 1. Copy environment file
```bash
cp .env.example .env
```

### 2. Start the database
```bash
make db-up
# Or: docker compose up -d
```

### 3. Initialize the schema
```bash
make setup
# Or: python scripts/setup_db.py
```

That's it! Your database is running at `localhost:5432`.

## Common Commands

| Command | Description |
|---------|-------------|
| `make db-up` | Start PostgreSQL container |
| `make db-down` | Stop PostgreSQL container |
| `make db-reset` | Delete all data and recreate schema |
| `make db-shell` | Open PostgreSQL command line |
| `make db-logs` | View database logs |
| `make pgadmin` | Start with pgAdmin web UI |

## Database Access

**Connection Details:**
- Host: `localhost`
- Port: `5432`
- Database: `chesslab`
- Username: `chesslab`
- Password: `chesslab_dev`

**Connection String:**
```
postgresql://chesslab:chesslab_dev@localhost:5432/chesslab
```

## Using pgAdmin (Optional)

For a visual database management interface:

```bash
make pgadmin
```

Then open http://localhost:5050 in your browser:
- Email: `admin@chesslab.local`
- Password: `admin`

To connect to the database in pgAdmin:
1. Right-click "Servers" → "Register" → "Server"
2. Name: `ChessLab`
3. Connection tab:
   - Host: `postgres` (when pgAdmin is in Docker) or `localhost`
   - Port: `5432`
   - Database: `chesslab`
   - Username: `chesslab`
   - Password: `chesslab_dev`

## Troubleshooting

### Port 5432 already in use
If you have local PostgreSQL running:
```bash
# Stop local PostgreSQL (Windows)
net stop postgresql-x64-14

# Or change the port in docker-compose.yml
ports:
  - "5433:5432"  # Use port 5433 instead
```

### WSL Issues
Docker Desktop on Windows with WSL2 should work out of the box. Make sure:
1. Docker Desktop is running
2. WSL2 integration is enabled in Docker Desktop settings
3. You're running commands from your WSL2 terminal

### Database won't start
```bash
# Check logs
make db-logs

# Clean restart
docker compose down -v
make db-up
```

## Data Persistence

Database data is stored in a Docker volume called `postgres_data`. It persists between container restarts.

To completely remove all data:
```bash
docker compose down -v
```

## Production Notes

For production, you should:
1. Use strong passwords (change in docker-compose.yml)
2. Use managed database services (AWS RDS, Google Cloud SQL, etc.)
3. Set up proper backups
4. Use SSL connections
5. Configure environment-specific credentials
