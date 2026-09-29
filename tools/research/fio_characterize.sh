#!/usr/bin/env bash
# Device I/O characterization sweep (bs x jobs x iodepth) via fio, writing a
# TSV results file. Ported from ~/research/fio/run.sh on the isfcr benchmark
# host, generalized to take the target file/device and output path as args
# instead of a hard-coded home path.
#
# Usage:
#   tools/research/fio_characterize.sh <target-file-or-device> [output.tsv]
#
# Example (characterize the drive underneath a built index, read-only):
#   tools/research/fio_characterize.sh \
#     indices/PipeANN/bigann-10m/R64_L128_pq32/pipeann_disk.index \
#     results/fio/bigann10m_nvme.tsv
#
# Requires: fio with io_uring engine support, python3.
set -euo pipefail

TARGET="${1:?usage: fio_characterize.sh <target-file-or-device> [output.tsv]}"
OUT="${2:-results/fio/$(basename "$TARGET")_$(date +%Y%m%d_%H%M%S).tsv}"
mkdir -p "$(dirname "$OUT")"

if [ ! -e "$TARGET" ]; then
  echo "error: target '$TARGET' does not exist" >&2
  exit 1
fi

TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT

echo -e "bs\tjobs\tqd\tiops\tMBps\tlat_mean_us\tlat_p99_us\tcpu_usr\tcpu_sys" > "$OUT"

for bs in 512 4k 8k 16k 32k; do
  for cfg in "1 1" "1 4" "1 16" "1 64" "4 16" "4 64" "8 64"; do
    # shellcheck disable=SC2086
    set -- $cfg
    jobs="$1"
    qd="$2"
    fio --name=r --filename="$TARGET" --readonly --direct=1 --ioengine=io_uring \
        --rw=randread --bs="$bs" --iodepth="$qd" --numjobs="$jobs" \
        --group_reporting --time_based --runtime=8 --ramp_time=1 \
        --output-format=json > "$TMP_JSON" 2>/dev/null
    python3 -c "
import json
d = json.load(open('$TMP_JSON'))['jobs'][0]
r = d['read']
print('$bs\t$jobs\t$qd\t%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f' % (
    r['iops'], r['bw'] / 1024, r['clat_ns']['mean'] / 1e3,
    r['clat_ns']['percentile']['99.000000'] / 1e3, d['usr_cpu'], d['sys_cpu']))
" >> "$OUT"
  done
done

echo "DONE" >> "$OUT"
echo "wrote $OUT" >&2
