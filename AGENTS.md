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
sudo -E env PYTHONPATH="$PYTHONPATH" uv run ann-suite run --config configs/example.yaml  # Run benchmark sudo and env needed for eBPF
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
  - `monitoring/`: `ResourceMonitor` (Docker stats), `CgroupsV2Collector` (enhanced I/O metrics)
  - `datasets/`: Download and load HDF5/NumPy datasets
  - `results/storage.py`: JSON/CSV result persistence
- **library/algorithms/**: Each algorithm has Dockerfile + `runner.py`; use `base_runner.py` utilities
- **configs/**: YAML benchmark configs with parameter sweeps (list values auto-expand)

## Key Patterns

- **Container protocol**: Algorithms receive JSON config via `--mode build|search --config '{...}'`, output JSON to stdout or `/results/metrics.json`
- **Metrics hierarchy**: `BenchmarkResult` contains structured `CPUMetrics`, `MemoryMetrics`, `DiskIOMetrics`, `LatencyMetrics`
- **Parameter sweeps**: List values in `search.args` expand via `itertools.product`
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
