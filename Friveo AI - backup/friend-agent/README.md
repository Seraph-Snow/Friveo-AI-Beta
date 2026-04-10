# Friend Agent — Setup Guide

## Prerequisites (install these on your machine)
- **Docker Desktop** — https://www.docker.com/products/docker-desktop
- **VS Code** — https://code.visualstudio.com
- **VS Code Extensions**: Docker, Python, Pylance, ESLint

## First Time Setup

### 1. Clone and configure
```bash
git clone <your-repo>
cd friend-agent
cp .env.example .env     # edit .env with your own passwords
```

### 2. Start all services
```bash
make up
# Wait ~60 seconds for everything to start
make ps   # all services should show "healthy" or "running"
```

### 3. Pull LLM models (one time, ~2.5GB download)
```bash
make pull-models
```

### 4. Set up Langfuse (one time)
1. Open http://localhost:3000
2. Log in with credentials from your .env (LANGFUSE_EMAIL / LANGFUSE_PASSWORD)
3. Create a new project called "friend-agent"
4. Go to Settings → API Keys → Create new key
5. Copy the public and secret keys into your .env:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
   LANGFUSE_SECRET_KEY=sk-lf-xxxxx
   ```
6. Restart backend: `docker-compose restart backend`

### 5. Run database migrations
```bash
make migrate
```

## Daily Development
```bash
make up         # start everything
make logs       # watch all logs
make down       # stop everything (data preserved)
```

## Architecture Overview
```
Frontend (React)  →  Backend (FastAPI)  →  Agent (LangGraph)
                                        →  PostgreSQL (users, personality)
                                        →  MongoDB (sessions, events, mood)
                                        →  Qdrant (vector memory)
                                        →  Redis (cache + Celery broker)
                                        →  Ollama (local LLM)
                  →  Langfuse (trace every LLM call)
                  →  Flower (monitor Celery tasks)
```

## Service URLs
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:5173 | Chat UI |
| Backend API | http://localhost:8000 | REST + WebSocket |
| API Docs | http://localhost:8000/docs | Auto-generated Swagger |
| Langfuse | http://localhost:3000 | LLM trace dashboard |
| Flower | http://localhost:5555 | Celery task monitor |
| Qdrant | http://localhost:6333/dashboard | Vector DB browser |
