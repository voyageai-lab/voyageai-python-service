# VoyageAI Python Service

AI-powered travel itinerary generation service using FastAPI and OpenAI.

## Features

- **Structured Output**: Generates well-structured JSON itineraries using OpenAI's structured output mode
- **Pydantic Validation**: Strong type checking and validation for all data models
- **FastAPI**: Modern, fast web framework with automatic OpenAPI documentation

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
cd voyageai-python-service

# Install dependencies with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your OpenAI API key
```

### Running the Service

```bash
# Development mode
uvicorn voyageai.main:app --reload --port 8000

# Production mode
uvicorn voyageai.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/generate` | POST | Generate travel itinerary |

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=voyageai
```

## Project Structure

```
voyageai-python-service/
├── src/voyageai/
│   ├── config.py          # Application settings
│   ├── main.py            # FastAPI application
│   ├── schemas/           # Pydantic models
│   ├── services/          # Business logic
│   ├── routers/           # API routes
│   └── prompts/           # Prompt templates
├── tests/                 # Test files
└── docs/                  # Documentation
```

## License

MIT

