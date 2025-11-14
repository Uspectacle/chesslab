.PHONY: help db-up db-down db-reset db-shell db-logs setup

help:
	@echo "ChessLab Database Commands"
	@echo "=========================="
	@echo "make db-up       - Start PostgreSQL container"
	@echo "make db-down     - Stop PostgreSQL container"
	@echo "make db-reset    - Reset database (delete all data)"
	@echo "make db-shell    - Open PostgreSQL shell"
	@echo "make db-logs     - View database logs"
	@echo "make setup       - Initialize database schema"
	@echo "make pgadmin     - Start with pgAdmin UI"

db-up:
	docker compose up -d postgres
	@echo "Waiting for database to be ready..."
	@sleep 3
	@echo "Database is ready at localhost:5432"

db-down:
	docker compose down

db-reset:
	@echo "This will delete ALL data. Press Ctrl+C to cancel..."
	@sleep 3
	docker compose down -v
	docker compose up -d postgres
	@sleep 3
	python scripts/setup_db.py

db-shell:
	docker compose exec postgres psql -U chesslab -d chesslab

db-logs:
	docker compose logs -f postgres

setup: db-up
	@sleep 2
	python scripts/setup_db.py

pgadmin:
	docker compose --profile tools up -d
	@echo "pgAdmin available at http://localhost:5050"
	@echo "Email: admin@chesslab.local"
	@echo "Password: admin"
