# Evaluation-practice survey tools (analysis paper, claim S1)

1. `papers.txt`: `key|pdf url` per paper.
2. Download each PDF into its own empty directory (they are untrusted files).
3. `uv run --with pypdf python -I extract.py PDFDIR TXTDIR`: full text per paper (also writes a looser evidence file).
4. `python -I extract2.py TXTDIR EVDIR`: per paper and variable, the sentences matching a strict pattern (drive, device state, page cache, repeats/variance, query count, memory matching, load, run order).

Coding is from those sentences by a person; "none found" means no matching sentence. See the vault note `analysis-paper/Experiments/AP-S1-evaluation-practice-survey`.
