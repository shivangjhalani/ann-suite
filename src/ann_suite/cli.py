"""CLI for ANN Benchmarking Suite.

Provides a rich command-line interface using Typer for:
- Running benchmarks
- Generating reports
- Managing datasets
- Building algorithm containers
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ann_suite.core.config import load_config
from ann_suite.core.schemas import BenchmarkConfig, BenchmarkResult
from ann_suite.datasets.download import download_dataset
from ann_suite.evaluator import BenchmarkEvaluator
from ann_suite.results.storage import ResultsStorage
from ann_suite.runners.container_runner import ContainerRunner
from ann_suite.utils.logging import setup_logging

app = typer.Typer(
    name="ann-suite",
    help="ANN Benchmarking Suite",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to benchmark configuration file (YAML/JSON)"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory for results (overrides config)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    log_file: Path | None = typer.Option(
        None, "--log-file", help="Write logs to file in addition to console"
    ),
    json_logs: bool = typer.Option(
        False, "--json-logs", help="Output logs in JSON format (for programmatic parsing)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running"),
    include_raw_samples: bool = typer.Option(
        False,
        "--include-raw-samples",
        help="Also include raw samples in results_detailed.json (debug JSONL always stored)",
    ),
) -> None:
    """Run a benchmark suite from a configuration file."""
    setup_logging(
        level=log_level, log_file=log_file, json_format=json_logs, rich_console=not json_logs
    )

    # Load and validate configuration
    console.print(f"[bold blue]Loading configuration from {config}[/]")

    try:
        benchmark_config = load_config(config)
    except Exception as e:
        console.print(f"[bold red]Error loading config: {e}[/]")
        raise typer.Exit(1) from e

    # Override output directory if specified
    if output_dir is not None:
        benchmark_config.results_dir = output_dir

    # Override include_raw_samples if specified via CLI
    if include_raw_samples:
        benchmark_config.include_raw_samples = True

    # Show configuration summary
    _show_config_summary(benchmark_config)

    if dry_run:
        console.print("[bold green]Configuration is valid![/]")
        return

    # Run benchmarks
    console.print("[bold blue]Starting benchmark suite...[/]")
    evaluator = BenchmarkEvaluator(benchmark_config)

    try:
        results = evaluator.run()

        if results:
            console.print(f"\n[bold green]Completed {len(results)} benchmarks[/]")
            _show_results_table(results)
        else:
            console.print("[bold yellow]No results generated[/]")

    finally:
        evaluator.cleanup()


@app.command()
def report(
    results_dir: Path = typer.Option(
        Path("./results"), "--results", "-r", help="Results directory"
    ),
    run_name: str | None = typer.Option(None, "--run", help="Specific run to report on"),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, csv"
    ),
) -> None:
    """Generate a report from benchmark results."""
    storage = ResultsStorage(results_dir)

    try:
        results = storage.load(run_name)
    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/]")
        raise typer.Exit(1) from e

    if output_format == "table":
        _show_results_table(results)
    elif output_format == "json":
        import json

        data = [r.to_summary_dict() for r in results]
        console.print(json.dumps(data, indent=2, default=str))
    elif output_format == "csv":
        df = storage.load_dataframe(run_name)
        console.print(df.to_csv(index=False))


@app.command()
def build(
    algorithm: str = typer.Option(..., "--algorithm", "-a", help="Algorithm name to build"),
    algorithms_dir: Path = typer.Option(
        Path("./library/algorithms"), "--algorithms-dir", "-d", help="Path to algorithms directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild (no cache)"),
) -> None:
    """Build a Docker image for a specific algorithm."""

    setup_logging(level="INFO")

    algo_dir = algorithms_dir / algorithm.lower()
    if not algo_dir.exists():
        console.print(f"[bold red]Algorithm directory not found: {algo_dir}[/]")
        raise typer.Exit(1)

    dockerfile = algo_dir / "Dockerfile"
    if not dockerfile.exists():
        console.print(f"[bold red]Dockerfile not found: {dockerfile}[/]")
        raise typer.Exit(1)

    image_tag = f"ann-suite/{algorithm.lower()}:latest"
    console.print(f"[bold blue]Building image {image_tag}...[/]")

    # Create simplified runner just for building
    runner = ContainerRunner(
        data_dir=Path("./data"),
        index_dir=Path("./indices"),
        results_dir=Path("./results"),
    )

    success = runner.build_image(
        dockerfile_path=dockerfile,
        image_tag=image_tag,
        build_args={"NO_CACHE": "true"} if force else None,
        context_path=algorithms_dir,
    )

    if success:
        console.print(f"[bold green]Successfully built {image_tag}[/]")
    else:
        console.print(f"[bold red]Failed to build {image_tag}[/]")
        raise typer.Exit(1)


@app.command()
def download(
    dataset: str | None = typer.Option(None, help="Dataset name to download"),
    output: Path | None = typer.Option(None, help="Output directory (default: library/datasets/)"),
    list_datasets: bool = typer.Option(False, "--list", help="List available datasets"),
    quiet: bool = typer.Option(False, help="Suppress output"),
) -> None:
    """Download and prepare datasets used by the benchmark."""
    from ann_suite.datasets.download import list_datasets as list_ds

    if list_datasets:
        list_ds()
        return

    if dataset is None:
        console.print("[bold red]Error:[/] Missing option '--dataset'.")
        raise typer.Exit(1)

    if output is None:
        # Default to library/datasets relative to CWD if not specified
        output = Path("library/datasets")

    try:
        download_dataset(name=dataset, output_dir=output, quiet=quiet)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from None


@app.command()
def init_config(
    output: Path = typer.Option(
        Path("config.yaml"), "--output", "-o", help="Output configuration file"
    ),
) -> None:
    """Generate a sample configuration file."""
    sample_config = """\
# ANN Benchmarking Suite Configuration
name: "HNSW vs DiskANN Benchmark"
description: "Comparing In-Memory HNSW with Disk-Based DiskANN"

# Directories
data_dir: "./library/datasets"
results_dir: "./results"
index_dir: "./indices"

# Resource monitoring interval (ms)
monitor_interval_ms: 100

# Algorithms to benchmark
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
    algorithm_type: memory
    build:
      timeout_seconds: 1800
      args:
        M: 16
        ef_construction: 200
        num_threads: 4
    search:
      timeout_seconds: 300
      k: 10
      args:
        ef: 100

  - name: DiskANN
    docker_image: ann-suite/diskann:latest
    algorithm_type: disk
    build:
      timeout_seconds: 3600
      args:
        R: 64
        L: 100
        alpha: 1.2
        num_threads: 4
        # ADVISORY memory hints (NOT hard limits!) - for index optimization
        build_memory_maximum: 2.0   # Target memory budget (GB), actual will be higher
        # search_memory_maximum is a BUILD arg for index layout optimization
        search_memory_maximum: 0.5  # Target search-time memory budget (GB)
    search:
      timeout_seconds: 300
      k: 10
      args:
        Ls: 100

# Datasets to benchmark on
# Download with: uv run ann-suite download --dataset sift-10k
datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    query_path: sift-10k/queries.npy
    ground_truth_path: sift-10k/ground_truth.npy
    distance_metric: L2
    dimension: 128
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(sample_config)
    console.print(f"[bold green]Sample configuration written to {output}[/]")


def _show_config_summary(config: BenchmarkConfig) -> None:
    """Display a summary of the benchmark configuration."""
    table = Table(title="Benchmark Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Name", config.name)
    table.add_row("Algorithms", str(len(config.enabled_algorithms)))
    table.add_row("Datasets", str(len(config.datasets)))
    table.add_row("Data Dir", str(config.data_dir))
    table.add_row("Results Dir", str(config.results_dir))

    console.print(table)

    # Algorithm details
    if config.enabled_algorithms:
        algo_table = Table(title="Algorithms")
        algo_table.add_column("Name", style="cyan")
        algo_table.add_column("Image", style="white")
        algo_table.add_column("Type", style="green")

        for algo in config.enabled_algorithms:
            algo_table.add_row(algo.name, algo.docker_image, algo.algorithm_type.value)

        console.print(algo_table)


def _show_results_table(results: list[BenchmarkResult]) -> None:
    """Display benchmark results with detailed metrics."""
    from rich.panel import Panel

    for r in results:
        # Main results table
        main_table = Table(show_header=False, box=None, padding=(0, 2))
        main_table.add_column("Metric", style="dim")
        main_table.add_column("Value", style="bold")

        main_table.add_row("Algorithm", f"[cyan]{r.algorithm}[/]")
        main_table.add_row("Dataset", r.dataset)
        main_table.add_row("Recall@k", f"[green]{r.recall:.4f}[/]" if r.recall else "N/A")
        main_table.add_row("QPS", f"[green]{r.qps:,.1f}[/]" if r.qps else "N/A")
        main_table.add_row(
            "Build Time",
            f"{r.total_build_time_seconds:.2f}s" if r.total_build_time_seconds else "N/A",
        )

        # Latency metrics
        lat = r.latency
        main_table.add_row("", "")
        main_table.add_row("[yellow]Latency[/]", "")
        main_table.add_row("  Mean", f"{lat.mean_ms:.3f} ms" if lat.mean_ms > 0 else "N/A")
        main_table.add_row("  P50", f"{lat.p50_ms:.3f} ms" if lat.p50_ms > 0 else "N/A")
        main_table.add_row("  P95", f"{lat.p95_ms:.3f} ms" if lat.p95_ms > 0 else "N/A")
        main_table.add_row("  P99", f"[yellow]{lat.p99_ms:.3f} ms[/]" if lat.p99_ms > 0 else "N/A")

        # Resource metrics (phase-separated)
        main_table.add_row("", "")
        main_table.add_row("[magenta]Resources[/]", "")
        main_table.add_row(
            "  Build Peak RAM",
            f"{r.memory.build_peak_rss_mb:.1f} MB" if r.memory.build_peak_rss_mb > 0 else "N/A",
        )
        main_table.add_row(
            "  Search Peak RAM",
            f"{r.memory.search_peak_rss_mb:.1f} MB" if r.memory.search_peak_rss_mb > 0 else "N/A",
        )
        main_table.add_row(
            "  Search Avg RAM",
            f"{r.memory.search_avg_rss_mb:.1f} MB" if r.memory.search_avg_rss_mb > 0 else "N/A",
        )
        main_table.add_row(
            "  Search Avg CPU",
            f"{r.cpu.search_avg_cpu_percent:.1f}%" if r.cpu.search_avg_cpu_percent > 0 else "N/A",
        )
        main_table.add_row(
            "  Search CPU/Query",
            f"{r.cpu.search_cpu_time_per_query_ms:.3f} ms"
            if r.cpu.search_cpu_time_per_query_ms > 0
            else "N/A",
        )

        # Disk I/O metrics (CRITICAL) - Search phase metrics
        dio = r.disk_io
        if dio.search_avg_read_iops > 0 or dio.search_avg_write_iops > 0:
            main_table.add_row("", "")
            main_table.add_row("[red]Disk I/O (Search Phase)[/]", "")
            main_table.add_row("  Read IOPS", f"{dio.search_avg_read_iops:.1f}")
            main_table.add_row("  Write IOPS", f"{dio.search_avg_write_iops:.1f}")
            main_table.add_row("  Read MB/s", f"{dio.search_avg_read_throughput_mbps:.1f}")
            main_table.add_row("  Pages Read (4KB)", f"{dio.search_total_pages_read:,}")
            if dio.search_pages_per_query:
                main_table.add_row("  Pages/Query", f"{dio.search_pages_per_query:.1f}")
            if dio.warmup_read_mb > 0:
                main_table.add_row("  Warmup Read", f"{dio.warmup_read_mb:.1f} MB")

        # Hyperparameters
        if r.hyperparameters:
            main_table.add_row("", "")
            main_table.add_row("[dim]Hyperparameters[/]", "")
            if "build" in r.hyperparameters:
                for k, v in r.hyperparameters["build"].items():
                    main_table.add_row(f"  build.{k}", str(v))
            if "search" in r.hyperparameters:
                for k, v in r.hyperparameters["search"].items():
                    main_table.add_row(f"  search.{k}", str(v))

        console.print(
            Panel(main_table, title=f"[bold]{r.algorithm} on {r.dataset}[/]", border_style="blue")
        )
        console.print()


if __name__ == "__main__":
    app()
