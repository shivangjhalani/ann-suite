# DiskANN Algorithm Container for ANN Benchmarking Suite
#
# Uses diskannpy built FROM SOURCE at the upstream commit matching the 0.7.0
# release, with a small patch that accumulates the library's native QueryStats
# (I/O count, bytes read, distance computations, hops, cache hits) across
# searches and exposes them via index.get_search_stats()/reset_search_stats().
# See patches/stats.patch.
# Reference: https://github.com/microsoft/DiskANN

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

LABEL maintainer="ANN Suite Contributors"
LABEL description="DiskANN algorithm using instrumented diskannpy (source build)"
LABEL version="2.0.0"

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies: toolchain, libaio (async I/O reader), git/patch for source build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libaio-dev \
    libboost-program-options-dev \
    git \
    patch \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Intel MKL 2020.4 (required by the DiskANN core; same version as upstream CI).
# trusted=yes: the 2019-era repo key fails modern APT (sequoia) verification;
# the channel is HTTPS and the package version is pinned.
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/mkl all main" \
    > /etc/apt/sources.list.d/intel-mkl.list && \
    apt-get update && apt-get install -y --no-install-recommends intel-mkl-64bit-2020.4-912 && \
    rm -rf /var/lib/apt/lists/*

# Fetch pristine DiskANN at the commit matching diskannpy 0.7.0 and apply the
# search-statistics patch
COPY diskann/patches/stats.patch /tmp/diskann-stats.patch
RUN wget -q https://github.com/microsoft/DiskANN/archive/cce54e50a493ed76a446b6d7f7d82193faafe115.tar.gz \
        -O /tmp/diskann.tar.gz && \
    cd /tmp && tar xzf diskann.tar.gz && \
    cd /tmp/DiskANN-cce54e50a493ed76a446b6d7f7d82193faafe115 && \
    patch -p1 < /tmp/diskann-stats.patch

# Build instrumented diskannpy from source
# MKL paths follow the Debian package layout under /opt/intel/mkl
RUN uv pip install --system --no-cache "numpy==1.25" cmake==3.27.9 ninja pybind11==2.11.1 setuptools wheel && \
    CMAKE_ARGS="-DPYBIND=ON -DOMP_PATH=/opt/intel/mkl/lib/intel64 -DMKL_PATH=/opt/intel/mkl/lib/intel64 -DMKL_INCLUDE_PATH=/opt/intel/mkl/include" \
    uv pip install --system --no-cache /tmp/DiskANN-cce54e50a493ed76a446b6d7f7d82193faafe115

# Copy algorithm code and utilities
COPY utils.py .
COPY diskann/algorithm/ ./algorithm/

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_THREADING_LAYER=INTEL

# Default entrypoint
ENTRYPOINT ["python", "-m", "algorithm.runner"]

# Volume mounts for data and results
# CRITICAL: DiskANN must write indices to /data/index/ for accurate I/O metrics
VOLUME ["/data", "/results"]
