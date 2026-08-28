# HNSW Algorithm Container for ANN Benchmarking Suite
#
# Uses hnswlib built FROM SOURCE with a small instrumentation patch that enables
# the library's built-in search metrics (distance computations, hops) on the
# layer-0 search path, adds a candidates-explored counter, and exposes them via
# index.get_search_stats()/reset_search_stats(). See patches/stats.patch.
# Reference: https://github.com/nmslib/hnswlib

FROM python:3.12-slim

# Install uv for fast, modern dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

LABEL maintainer="ANN Suite Contributors"
LABEL description="HNSW algorithm using instrumented hnswlib (source build)"
LABEL version="2.0.0"

WORKDIR /app

# Install build dependencies (g++ needed for hnswlib C++ extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    patch \
    && rm -rf /var/lib/apt/lists/*

# Install build prerequisites
COPY hnsw/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt \
    pybind11==2.11.1 setuptools wheel

# Fetch pristine hnswlib 0.8.0 sdist and apply the search-statistics patch
COPY hnsw/patches/stats.patch /tmp/hnswlib-stats.patch
RUN pip download --no-binary :all: --no-deps --no-build-isolation hnswlib==0.8.0 -d /tmp/src && \
    cd /tmp/src && tar xzf hnswlib-0.8.0.tar.gz && \
    cd hnswlib-0.8.0 && patch -p1 < /tmp/hnswlib-stats.patch && \
    uv pip install --system --no-cache .

# Copy algorithm code and utilities
COPY utils.py .
COPY hnsw/algorithm/ ./algorithm/

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["python", "-m", "algorithm.runner"]

# Volume mounts for data and results
VOLUME ["/data", "/results"]
