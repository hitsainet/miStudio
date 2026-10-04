# MechInterp Studio - Backend

Backend API for MechInterp Studio, an edge-deployed mechanistic interpretability platform.

## Technology Stack

- **Framework**: FastAPI 0.104+
- **Database**: PostgreSQL 14+ with SQLAlchemy 2.0+ (async)
- **Cache/Queue**: Redis 7+ with Celery 5.3+
- **ML/AI**: PyTorch 2.1+, HuggingFace Transformers 4.35+, Datasets 2.14+
- **Python**: 3.10+

## Project Structure

```
backend/
├── src/
│   ├── api/              # FastAPI routes
│   ├── models/           # SQLAlchemy models
│   ├── services/         # Business logic
│   ├── workers/          # Celery tasks
│   ├── core/             # Core configuration (database, config, celery)
│   └── utils/            # Utility functions
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── alembic/              # Database migrations
└── pyproject.toml        # Dependencies and configuration
```

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Redis 7+
- Poetry (Python package manager)

### Installation

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate virtual environment:
```bash
poetry shell
```

4. Copy environment file and configure:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run database migrations:
```bash
alembic upgrade head
```

## Development

### Running the API server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Celery worker

```bash
celery -A src.core.celery_app worker --loglevel=info
```

### Running tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit -m unit

# Integration tests only
pytest tests/integration -m integration

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

See `.env.example` for required environment variables.

## License

See LICENSE file in project root.
