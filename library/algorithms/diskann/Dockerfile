# DiskANN Algorithm Container for ANN Benchmarking Suite
#
# Uses diskannpy - Microsoft's official Python bindings for DiskANN
# Reference: https://github.com/microsoft/DiskANN

# Using Python 3.11 because diskannpy has prebuilt wheels for 3.9-3.11
FROM python:3.11-slim

# Install uv for fast, modern dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

LABEL maintainer="ANN Suite Contributors"
LABEL description="DiskANN algorithm using diskannpy"
LABEL version="1.0.0"

WORKDIR /app

# Install Python dependencies
COPY diskann/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy algorithm code
# Copy utilities
COPY utils.py .

# Copy algorithm code
COPY diskann/algorithm/ ./algorithm/

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Default entrypoint
ENTRYPOINT ["python", "-m", "algorithm.runner"]

# Volume mounts for data and results
# CRITICAL: DiskANN must write indices to /data/index/ for accurate I/O metrics
VOLUME ["/data", "/results"]
