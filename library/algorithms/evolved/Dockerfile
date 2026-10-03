# Evolved disk-ANN program container for ANN Suite (OpenEvolve candidates).
#
# The image holds only the fixed harness/runner plus the building blocks
# candidates may use (numpy, faiss for k-means/PQ/IVF, numba for custom kernels,
# diskannpy for Vamana graphs). The candidate program itself is not baked in: it
# is passed per build as build_args.program (a file under /data), so a new
# candidate needs no image rebuild. See algorithm/harness.py for its interface.
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
LABEL maintainer="ANN Suite Contributors"
LABEL description="Harness for evolved disk-ANN programs"
LABEL version="1.0.0"
WORKDIR /app
COPY evolved/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY evolved/algorithm/ ./algorithm/
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# numba caches compiled kernels next to the source by default; keep them in /tmp.
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENTRYPOINT ["python", "-m", "algorithm.runner"]
VOLUME ["/data", "/results"]
