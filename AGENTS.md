# ANN Suite - Agent Guidance

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest                        # Run all tests
uv run pytest tests/test_schemas.py  # Run single test file
uv run pytest -k "test_valid_config" # Run tests matching pattern
uv run ruff check src tests          # Lint
uv run ruff format src tests         # Format
uv run mypy src                      # Type check (strict mode)
uv run ann-suite --help              # CLI entrypoint
uv run ann-suite build --algorithm HNSW  # Build algorithm container
uv run ann-suite run --config configs/example.yaml
```

## Requirements

- **cgroups v2 is required** for metrics collection. The suite will fail at startup if cgroups v2 is not available.
- Verify with: `cat /sys/fs/cgroup/cgroup.controllers`
- See `docs/METRICS.md` for setup instructions if cgroups v2 is not enabled.

## Architecture

- **src/ann_suite/**: Main package
  - `cli.py`: Typer CLI with Rich console output
  - `evaluator.py`: Orchestrates benchmark pipeline (dataset→build→search→aggregate)
  - `core/schemas.py`: Pydantic models - key types: `BenchmarkConfig`, `AlgorithmConfig`, `DatasetConfig`, `BenchmarkResult`, `ContainerProtocol`
  - `core/config.py`: YAML/JSON config loading with validation
  - `runners/container_runner.py`: Docker lifecycle (volumes mount to `/data`, `/data/index`, `/results`)
  - `monitoring/`: `CgroupsV2Collector` reads the container's cgroup v2 files (CPU, memory, io.stat, PSI); `base.py` has collector types and device-level readers
  - `datasets/`: Download and load HDF5/NumPy datasets
  - `results/storage.py`: JSON/CSV result persistence
- **library/algorithms/**: Each algorithm has a Dockerfile + `algorithm/runner.py`; shared helpers (recall, latency percentiles) live in `library/algorithms/utils.py`. Variant images (`Dockerfile.hashfix`, `.visittrace`, `.cachedump`) apply extra DiskANN patches
- **configs/**: current benchmark configs; `configs/archive/` holds configs of past experiments (cited from the research vault)
- **tools/**: result analysis scripts; `tools/research/` holds offline research analyses (navigability, co-location, dataset prep)

## Key Patterns

- **Container protocol**: Algorithms receive JSON config via `--mode build|search --config '{...}'`, output JSON to stdout or `/results/metrics.json`
- **Metrics hierarchy**: `BenchmarkResult` contains structured `CPUMetrics`, `MemoryMetrics`, `DiskIOMetrics`, `LatencyMetrics`
- **Parameter sweeps**: list values in `build.args`/`search.args` expand via `itertools.product`; `search.sweep` gives explicit points. Each unique build is built once and reused by its search points
- **Fair comparison**: set the top-level `resources:` (memory_limit, cpu_affinity) so every algorithm gets the same budget; results record it in `run_conditions`
- **Cache reset**: every search point runs in a fresh container after an OS page-cache drop (`drop_caches_before`, default true; set `ANN_SUITE_SUDO_PASSWORD` when sudo needs a password). A failed drop fails the point
- **Error handling**: Return partial `BenchmarkResult` on failure; check `PhaseResult.success`

## Code Style

- Python 3.12+; always `from __future__ import annotations`
- Pydantic v2: use `Field(description=...)`, `field_validator`, `model_validator`
- Ruff: line-length=100, select=E,F,I,N,W,UP,B,C4,SIM; ignore=B008,N803,N806,E501
- Type hints required everywhere; `mypy --strict` must pass
- Use `pathlib.Path`, not strings; resolve paths with `.resolve()`
- Enums inherit `(str, Enum)` for JSON serialization
- Logging via `logging.getLogger(__name__)`; include run_id for correlation
- Tests: pytest, class-based grouping (`class TestAlgorithmConfig`), `-v --tb=short`
