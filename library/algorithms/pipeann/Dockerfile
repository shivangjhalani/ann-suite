# PipeANN algorithm container for ANN Suite.
#
# PipeANN (https://github.com/thustorage/PipeANN) is a disk-based ANN index
# (Vamana/DiskANN-style graph, PQ- or RaBitQ-compressed) with a pipelined
# search mode that overlaps SSD I/O with distance computation.
#
# Built from source at a pinned upstream commit, in READ_ONLY_TESTS +
# NO_MAPPING mode (search-only: no tag<->id mapping table, no update paths,
# no page-cache indirection - the recommended configuration for search
# benchmarking; see docs/cpp-interface.md "Build Configuration" upstream).
#
# The load-aware modifications applied on top of that commit come from
# ~/research/PipeANN (branch load-aware) on the isfcr benchmark host, a fork
# used there for latency-under-load experiments:
#   - patches/load_aware.patch: an optional env-controlled I/O width governor
#     (PIPEANN_QSTAR / PIPEANN_WMIN, see vendor/io_governor.h - unset means
#     stock PipeANN behavior) and PIPEANN_SQPOLL=0 to disable io_uring
#     SQPOLL per search thread (SQPOLL pins a CPU core per thread; with many
#     search threads this saturates the machine even when the workload is
#     I/O-bound - the researcher's benchmark note, confirmed on isfcr).
#   - vendor/io_governor.h, vendor/search_openloop.cpp: new files (not a diff
#     against the pinned commit - they don't exist upstream), copied in
#     verbatim. search_openloop.cpp is a standalone Poisson/closed arrival
#     driver (M/G/T FIFO queue, latency = completion - scheduled arrival) that
#     this suite's open-loop search mode invokes directly; see
#     algorithm/runner.py.
#
# Reference: https://github.com/thustorage/PipeANN

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

LABEL maintainer="ANN Suite Contributors"
LABEL description="PipeANN algorithm (pipelined disk-based ANN search), source build"
LABEL version="1.0.0"

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies. openblas/lapacke stand in for the intel-mkl-full-dev the
# upstream README suggests (no Intel apt repo dependency here - see the
# DiskANN Dockerfile for why that's worth avoiding). liburing is built from
# the vendored third_party/liburing source. libgoogle-perftools-dev provides
# tcmalloc (USE_TCMALLOC=ON is PipeANN's default). grpc/protobuf/rocksdb are
# NOT needed: those only gate BUILD_MILVUS_SERVER and the Python interface
# (BUILD_PYTHON_INTERFACE), both OFF by default, and we only build the plain
# C++ test/tool binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    patch \
    ca-certificates \
    libaio-dev \
    libgoogle-perftools-dev \
    libopenblas-dev \
    liblapacke-dev \
    libeigen3-dev \
    libboost-program-options-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG PIPEANN_COMMIT=b9ec4ba
RUN git clone https://github.com/thustorage/PipeANN.git /opt/pipeann && \
    cd /opt/pipeann && git checkout ${PIPEANN_COMMIT}

# Apply the load-aware patch (modifies 3 tracked upstream files) and add the
# two new files it depends on.
COPY pipeann/patches/load_aware.patch /tmp/pipeann-load-aware.patch
COPY pipeann/vendor/io_governor.h /opt/pipeann/include/utils/io_governor.h
COPY pipeann/vendor/search_openloop.cpp /opt/pipeann/tests/search_openloop.cpp
RUN cd /opt/pipeann && patch -p1 < /tmp/pipeann-load-aware.patch

# Build liburing (vendored under third_party/, not a git submodule).
RUN cd /opt/pipeann/third_party/liburing && ./configure && make -j"$(nproc)"

# Search-only build: disables the tag<->id mapping table and update paths for
# best search throughput/I-O-count accuracy (see module docstring above).
RUN mkdir -p /opt/pipeann/build && cd /opt/pipeann/build && \
    ADDITIONAL_DEFINITIONS="-DREAD_ONLY_TESTS -DNO_MAPPING" cmake .. \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_MILVUS_SERVER=OFF && \
    make -j"$(nproc)" build_disk_index build_memory_index search_disk_index search_openloop \
        gen_random_slice compute_groundtruth vecs_to_bin change_pts

ENV PIPEANN_BIN=/opt/pipeann/build/tests
ENV PIPEANN_UTIL_BIN=/opt/pipeann/build/tests/utils

# Runtime: only numpy is needed (dataset/query format conversion; the binaries
# themselves do all the search/build work and report their own recall/QPS/
# latency stats, so unlike DiskANN/SPANN this runner does not need
# utils.compute_recall).
COPY pipeann/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY pipeann/algorithm/ ./algorithm/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "algorithm.runner"]

VOLUME ["/data", "/results"]
