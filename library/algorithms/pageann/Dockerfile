# PageANN / LAANN algorithm container for ANN Suite.
#
# PageANN (arXiv:2509.25487) and LAANN (arXiv:2606.02784) share one codebase,
# a DiskANN fork (https://github.com/Dingyi-Kang/PageANN). Both group vectors
# into 4 KB page nodes with a merged page-level neighbor list; they differ in
# the on-disk page format and the search loop, so one image serves both and the
# runner picks the variant from build_args.variant ("pageann" | "laann").
#
# Built from source at a pinned commit, following the upstream Dockerfile
# (ubuntu:jammy + Intel MKL from the Ubuntu archive; LLVM libomp-dev provides
# the libiomp5.so that the CMake MKL setup looks for).
FROM ubuntu:jammy
LABEL maintainer="ANN Suite Contributors"
LABEL description="PageANN and LAANN (page-node disk graph ANN), source build"
LABEL version="1.0.0"
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libaio-dev \
    liburing-dev \
    libgoogle-perftools-dev \
    libunwind-dev \
    libboost-dev \
    libboost-program-options-dev \
    libmkl-full-dev \
    libomp-dev \
    python3 \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*
ARG PAGEANN_COMMIT=480f72a
RUN git clone https://github.com/Dingyi-Kang/PageANN.git /opt/pageann && \
    cd /opt/pageann && git checkout ${PAGEANN_COMMIT}
RUN cmake -S /opt/pageann -B /opt/pageann/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/pageann/build -j"$(nproc)" --target \
        build_vamana_disk_index generate_page_graph generate_laann_graph search_disk_index \
        build_pageann_nav_graph build_laann_nav_graph reorder_pages_by_frequency
ENV PAGEANN_BIN=/opt/pageann/build/apps
COPY pageann/algorithm/ ./algorithm/
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-m", "algorithm.runner"]
VOLUME ["/data", "/results"]
