# =============================================================================
# FRIEND AGENT — Makefile
# =============================================================================
# WHY A MAKEFILE?
#   Docker commands get long and hard to remember. The Makefile gives you
#   short aliases. Type `make up` instead of `docker-compose up -d`.
#   It also documents the commands your team needs to know.
# HOW TO USE: Just type `make <target>` in your terminal.
# =============================================================================

.PHONY: up down logs ps pull-models reset help

## Start all services in the background
up:
	docker-compose up -d
	@echo ""
	@echo "Services starting... check status with: make ps"
	@echo ""
	@echo "URLs:"
	@echo "  Frontend:         http://localhost:5173"
	@echo "  Backend API:      http://localhost:8000"
	@echo "  API Docs:         http://localhost:8000/docs"
	@echo "  Langfuse:         http://localhost:3000"
	@echo "  Flower (Celery):  http://localhost:5555"
	@echo "  Qdrant Dashboard: http://localhost:6333/dashboard"

## Stop all services (keeps data volumes)
down:
	docker-compose down

## Stop all services AND delete all data (fresh start)
reset:
	docker-compose down -v
	@echo "All data volumes deleted. Next 'make up' starts fresh."

## Show running services and their status
ps:
	docker-compose ps

## Follow logs for all services (Ctrl+C to stop)
logs:
	docker-compose logs -f

## Follow logs for a specific service: make logs-backend
logs-%:
	docker-compose logs -f $*

## Pull LLM models into Ollama (run once after first `make up`)
pull-models:
	@echo "Pulling llama3.2 (chat model ~2GB)..."
	docker exec -it friend-ollama ollama pull llama3.2
	@echo "Pulling nomic-embed-text (embedding model ~270MB)..."
	docker exec -it friend-ollama ollama pull nomic-embed-text
	@echo "Models ready!"

## Run database migrations (after schema changes)
migrate:
	docker exec -it friend-backend alembic upgrade head

## Open a shell inside the backend container
shell-backend:
	docker exec -it friend-backend bash

## Open a Postgres shell
shell-postgres:
	docker exec -it friend-postgres psql -U friendagent -d friendagent

## Open a MongoDB shell
shell-mongo:
	docker exec -it friend-mongo mongosh -u friendagent -p mongo_secret_change_me --authenticationDatabase admin friendagent

## Open a Redis CLI
shell-redis:
	docker exec -it friend-redis redis-cli

## Rebuild images (after requirements.txt changes)
rebuild:
	docker-compose build --no-cache backend frontend

help:
	@echo "Available commands:"
	@grep -E '^## ' Makefile | sed 's/## /  make /'
