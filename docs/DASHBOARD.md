# Visualization Dashboard

The suite ships a self-contained web dashboard for exploring and comparing
benchmark results across runs, algorithms, datasets, and hyperparameter sweeps.

## How it works

There are two pieces:

1. **`ann-suite dashboard`** — exports every run under `results/` into a single
   flat JSON file (`dashboard_data.json`), one row per
   algorithm × dataset × hyperparameter combination.
2. **`tools/dashboard/dashboard.html`** — a dependency-free page (Plotly.js via
   CDN) that loads that JSON and renders the chart grid.

No server, framework, or Python visualization dependency is required. The page
works in two ways:

- **Via `--open`:** `ann-suite dashboard --results ./results --open` starts a
  tiny local server and opens the page, which auto-fetches the JSON.
- **Manually:** open `tools/dashboard/dashboard.html` in a browser and use the
  **"Load data file…"** button to pick the generated `dashboard_data.json`.

## Metric organization

The dashboard uses a **one metric = one chart** layout, with a consistent control
chrome so every metric is read the same way:

- **Tabs by family:** Quality, Latency, CPU, Memory, Disk I/O, Build/Index.
- **Filters:** Run (experiment), dataset, and algorithm selectors.
- **X-axis:** switch between "grouped bars" (one x-tick per config) or a
  hyperparameter dimension (e.g. `search_ef`) to read parameter sweeps as lines.
- **Quality tab** also includes a **Recall vs QPS** scatter for the classic
  trade-off view.

## Custom algorithm metrics

Algorithms can emit arbitrary JSON in their container output (beyond the standard
`SearchOutput` / `BuildOutput` contract). These are captured automatically:

- `BenchmarkResult.custom_metrics` stores the non-standard keys, namespaced by
  phase (`search_*` / `build_*`).
- They are flattened into `custom_*` columns in the export (nested dicts become
  `custom_search_visited_mean`, lists are JSON-encoded with a `_raw` twin).
- The dashboard auto-discovers any `custom_*` columns present in the loaded data
  and renders one chart per metric.

Hyperparameters (including sweep dimensions) are likewise flattened into stable
`hp_*` columns (`hp_search_ef`, `hp_build_M`, `hp_k`), which is what makes the
x-axis selector able to plot sweeps.
