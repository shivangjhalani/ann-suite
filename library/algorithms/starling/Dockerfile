# Starling algorithm container for ANN Suite.
#
# Starling (SIGMOD'24, https://github.com/zilliztech/starling) is a DiskANN
# fork that relays out the Vamana disk graph so that graph-neighboring nodes
# share a 4 KB page (graph partition + index_relayout) and searches page by
# page ("page search"), optionally seeded by an in-memory navigation graph.
#
# Built from source at a pinned commit. patches/graph_partition_stream.patch
# comes from the isfcr research host's checkout (/home/isfcr/starling): the
# partitioner streams the disk index instead of loading it whole and keeps a
# single adjacency copy, which is what lets 100M-scale partitioning fit in RAM.
# It also replaces id2pid's unordered_map with a vector. The search path is
# unmodified upstream.
FROM ubuntu:jammy
LABEL maintainer="ANN Suite Contributors"
LABEL description="Starling (page-relayout disk graph ANN), source build"
LABEL version="1.0.0"
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    patch \
    ca-certificates \
    libaio-dev \
    libgoogle-perftools-dev \
    libunwind-dev \
    libboost-dev \
    libboost-program-options-dev \
    libmkl-full-dev \
    libomp-dev \
    python3 \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*
ARG STARLING_COMMIT=17dc3e8
RUN git config --global url."https://github.com/".insteadOf git@github.com: && \
    git clone https://github.com/zilliztech/starling.git /opt/starling && \
    cd /opt/starling && git checkout ${STARLING_COMMIT} && \
    git submodule update --init --recursive graph_partition
COPY starling/patches/graph_partition_stream.patch /tmp/gp.patch
RUN cd /opt/starling/graph_partition && patch -p1 < /tmp/gp.patch
RUN cmake -S /opt/starling -B /opt/starling/release -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/starling/release -j"$(nproc)"
ENV STARLING_BUILD=/opt/starling/release
COPY starling/algorithm/ ./algorithm/
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-m", "algorithm.runner"]
VOLUME ["/data", "/results"]
