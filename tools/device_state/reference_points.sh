#!/usr/bin/env bash
# Cross-session drift reference (analysis-paper hypotheses B1/B2): the same search points on the
# same untouched indexes, once per session, with device probes. Run from cron twice a day.
# Skips the session (does not wait) if any benchmark, search container or campaign tmux session is
# active, so it never overlaps a timed run. Output: results/reference_points/<YYYY-MM-DD_HHMM>/.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH
log=logs/reference_points.log
if pgrep -f "python.*(tools/device_state/|evolve_bench)|ann-suite run" >/dev/null \
  || [ -n "$(docker ps --format '{{.Names}}' | grep -v monitoring || true)" ] \
  || tmux ls 2>/dev/null | grep -qE '^(recheck|pvq|a2|span|rank|probe)'; then
  echo "$(date -Is) skipped: machine busy" >>"$log"
  exit 0
fi
out=results/reference_points/$(date +%F_%H%M)
base=/home/isfcr/evolve_indices/baselines
diskann=$(ls -d $base/DiskANN-B0.32/bigann10m-q2k/*/ | head -1)
pipeann=$(ls -d $base/PipeANN-pq32/bigann10m-q2k/*/ | head -1)
echo "$(date -Is) start $out" >>"$log"
# Both at recall ~0.95 on BIGANN-10M (2,000 queries), under the evolve baselines' 785 MB cap.
uv run python tools/device_state/probe_vs_qps.py --no-copy --out "$out" \
  --config configs/evolve/baselines/diskann.yaml --algo DiskANN-B0.32 --files '*' --original "$diskann" \
  --params '{"Ls": 40}' >>"$log" 2>&1
uv run python tools/device_state/probe_vs_qps.py --no-copy --out "$out" \
  --config configs/evolve/baselines/pipeann.yaml --algo PipeANN-pq32 --files '*' --original "$pipeann" \
  --params '{"mode": 2, "beam_width": 32, "Ls": 40}' >>"$log" 2>&1
echo "$(date -Is) done $out" >>"$log"
