# SPANN (Microsoft SPTAG) algorithm container for ANN Suite.
# SPANN keeps the centroid/head index in memory and reads posting lists from SSD.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

LABEL maintainer="ANN Suite Contributors"
LABEL description="SPANN algorithm using Microsoft's SPTAG implementation"
LABEL version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libaio-dev \
    libboost-all-dev \
    libnuma-dev \
    libtbb-dev \
    patch \
    && rm -rf /var/lib/apt/lists/*

# Pin the upstream source so benchmark images are reproducible. SPTAG includes
# the SPANN implementation under AnnService/{inc,src}/Core/SPANN.
ARG SPTAG_COMMIT=2ac3ebcab562bc81cdb8c7c98b35ea72f2703c3b
RUN git clone --recurse-submodules https://github.com/microsoft/SPTAG.git /opt/sptag \
    && cd /opt/sptag \
    && git checkout ${SPTAG_COMMIT} \
    && git submodule update --init --recursive \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPDK=OFF -DROCKSDB=OFF -DGPU=OFF \
    && cmake --build build --parallel 2 --target indexbuilder indexsearcher

COPY spann/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY utils.py .
COPY spann/algorithm/ ./algorithm/
ENV SPTAG_BIN=/opt/sptag/Release
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "algorithm.runner"]
VOLUME ["/data", "/results"]
