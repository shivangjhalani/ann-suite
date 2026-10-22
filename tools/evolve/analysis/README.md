# Offline analyses of built indexes (design v2 diagnosis, 2026-10-16)

CPU-only scripts that read a built `abl_fastscan_*` / `abl_v2*` index (mem/ + disk/lists.pages)
and the dataset's queries / ground truth. Run inside the `ann-suite/evolved` image with the
index and data mounted read-only, never while a build or a timed search is running.

| script | question |
|---|---|
| `diag_high_recall.py INDEX DATA [nq]` | where recall is lost: probe coverage, shortlist coverage, page order by fast-scan / refined / exact distance, vectors per page |
| `sim_code_split.py INDEX DATA L2\|IP m1:m2:opq ...` | first-stage code variants on the same lists: shortlist coverage, recall at 24/32 pages, fast-scan CPU |
| `lut_refine_check.py INDEX DATA` | table-form refinement (folded kappa) vs the per-candidate kernel; faiss fast-scan implementations; probe efSearch |
| `stage_cpu.py PROGRAM INDEX DATA METRIC np,sl,pages ...` | per-stage single-thread CPU of a program's Searcher |
| `locality_vs_code.py INDEX DATA L2\|IP NPROBE SHORTLIST [nq]` | pages needed with the locality layout vs a cluster-order proxy, for fast-scan-only, refined and exact page order (abl_v2 indexes) |
