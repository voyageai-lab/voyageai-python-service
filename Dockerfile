# Multi-stage build for VoyageAI Python AI Service
# Stage 1: Build with dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install system deps for confluent-kafka (librdkafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc librdkafka-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Install runtime deps for confluent-kafka
RUN apt-get update && apt-get install -y --no-install-recommends \
    librdkafka1 wget && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r voyageai && useradd -r -g voyageai voyageai

# Copy installed packages and source
COPY --from=builder /install /usr/local
COPY src ./src

# Health check (FastAPI server)
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:8000/api/v1/health || exit 1

USER voyageai

EXPOSE 8000

# Default: run FastAPI server
CMD ["uvicorn", "voyageai.main:app", "--host", "0.0.0.0", "--port", "8000"]
