# HNSW Algorithm Container for ANN Benchmarking Suite
#
# Uses hnswlib - the most popular pure HNSW implementation
# Reference: https://github.com/nmslib/hnswlib

FROM python:3.12-slim

# Install uv for fast, modern dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

LABEL maintainer="ANN Suite Contributors"
LABEL description="HNSW algorithm using hnswlib"
LABEL version="1.0.0"

WORKDIR /app

# Install build dependencies (g++ needed for hnswlib C++ extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY hnsw/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy algorithm code
# Copy utilities
COPY utils.py .

# Copy algorithm code
COPY hnsw/algorithm/ ./algorithm/

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["python", "-m", "algorithm.runner"]

# Volume mounts for data and results
VOLUME ["/data", "/results"]
