"""Pass 2 of the S1 coding: keyword evidence from the full text of each paper.
usage: python -I extract.py PDFDIR OUTDIR  (writes OUTDIR/<key>.txt text and <key>.evidence.md)"""

import re, sys
from pathlib import Path
from pypdf import PdfReader

pdfs, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
VARS = {
    "drive": r"\b(NVMe|SSD)s?\b[^.\n]{0,100}\b(Samsung|Intel|Western Digital|WD|Micron|Kioxia|Solidigm|SK ?hynix|Crucial|Optane|Seagate|Corsair|PM\d{3,4}|9[0-9]0 ?(PRO|EVO)|P\d{4}|DC ?P\d+|SN\d{3}|TB\b|GB\b)|\b(Samsung|Intel|Micron|Kioxia|Solidigm|Optane|Crucial|WD Black|PM9\w+|980 ?PRO|990 ?PRO|P5800X|P4510|P4610|P5510|D7-P\d+|CM\d|SN\d{3})\b",
    "state": r"precondition|steady.state|SLC.cache|pSLC|garbage.collection|over.?provision|wear|fill(ed)? level|fully (written|filled)|TRIM|device state|drive state|warm.?up",
    "pagecache": r"drop_caches|page.cache|O_DIRECT|direct.I/?O|buffered I/?O|clear(ed)? (the )?(OS |file ?system )?cache|flush(ed)? (the )?cache",
    "repeats": r"(repeat|averag|median|trial|run (each|the|\d)|runs?\b[^.]{0,40}times|times[^.]{0,30}(and|then) (average|report)|standard deviation|std\b|error bar|variance|confidence interval|best of|fluctuat|variab)",
    "queries": r"\d[\d,\.]*[ kKM]* (test |query |search )?queries|query (set|file|workload|log|trace)|queries (are|were) (sampled|drawn|randomly|taken)",
    "memory": r"memory (budget|limit|constraint|footprint|usage|cap)|DRAM (budget|limit|usage|footprint|consumption)|cgroup|memory-matched|same memory|equal memory|under the same memory",
    "load": r"closed.loop|open.loop|Poisson|arrival rate|concurrent (clients|queries|requests)|\d+ (query |search |client )?threads|number of threads|QPS",
    "interleave": r"interleav|random(ized|ised)? order|order of (the )?(experiments|runs|systems|baselines)|alternat(e|ing) (between|runs|the order)|RMIT",
}
for pdf in sorted(pdfs.glob("*.pdf")):
    k = pdf.stem
    try:
        text = "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)
    except Exception as e:  # noqa: BLE001
        print(k, "FAILED", e)
        continue
    (out / f"{k}.txt").write_text(text)
    flat = re.sub(r"\s+", " ", text)
    md = [f"# {k}  ({len(PdfReader(pdf).pages)} pages, {len(flat)} chars)\n"]
    for var, pat in VARS.items():
        hits = [m for m in re.finditer(pat, flat, re.I)]
        md.append(f"\n## {var}  ({len(hits)} hits)")
        last = -1000
        shown = 0
        for m in hits:
            if m.start() - last < 400:
                continue
            last = m.start()
            a, b = max(0, m.start() - 160), min(len(flat), m.end() + 200)
            md.append(f"- …{flat[a:b]}…")
            shown += 1
            if shown >= 6:
                break
    (out / f"{k}.evidence.md").write_text("\n".join(md))
    print(k, len(flat))
