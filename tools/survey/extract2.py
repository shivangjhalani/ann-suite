"""Pass 2 of the S1 coding, sentence level: for each paper and variable, the sentences that
match a strict pattern. usage: python -I extract2.py TXTDIR OUTDIR"""

import re, sys
from pathlib import Path

txt, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
V = {
    "drive": r"\b(SSD|NVMe|disk)\b.{0,120}\b(Samsung|Intel|WD|Western Digital|Micron|Kioxia|Solidigm|SK ?hynix|Crucial|Optane|Seagate|Corsair|PM\d{3,4}|\d{3} ?(PRO|EVO)|P\d{4}X?|SN\d{3}|local ssd|EBS|gp3|io2|Azure|i3en|i4i|c6id|n2-standard)|\b(Samsung|Intel|Micron|Kioxia|Solidigm|Optane|Crucial|PM9\w+|980 ?PRO|990 ?PRO|P5800X|P4510|P4610|P5510|P5530|D7-P\d+|SN8\d\d|SN7\d\d)\b",
    "state": r"precondition|steady.state|SLC.cache|pSLC|garbage.collect|over.?provision|wear.level|fill(ed)? level|fully (written|filled)|\bTRIM\b|device state|drive state|warm(ed)?.?up (the )?(ssd|drive|disk)|ssd (is|was) (warm|fresh)|write amplification",
    "pagecache": r"drop_caches|page.cache|O_DIRECT|o_direct|direct.I/?O|buffered I/?O|(clear|drop|flush)(ed|es)? (the )?(OS |file ?system |disk |page )?cache",
    "repeats": r"(repeat(ed)?|run|ran|execute[ds]?|measure[ds]?|average[ds]?|trial)s?\b[^.]{0,60}\b(\d+|three|five|ten|twice|multiple|several) (times|runs|trials|repetitions)|(standard deviation|error bars?|confidence interval|relative std|best (of|run)|median of|mean of \d+)",
    "queries": r"\b\d[\d,\.]*[ ]?[kKM]?\s+(test |query |search )?queries\b|queries (are|were) (sampled|drawn|randomly|taken|generated)|query (set|file|workload|log|trace) (of|with|contains|is)",
    "memory": r"memory (budget|limit|constraint|footprint|cap)|(same|equal|identical|matched|fixed|limit(ed)?) (the )?(DRAM|memory|RAM)|cgroup|memory-constrained|\d+ ?(GB|MB|GiB) (of )?(DRAM|memory|RAM) (budget|limit)",
    "load": r"closed.loop|open.loop|Poisson|arrival rate|concurrent (clients|queries|requests)|\b\d+ (query |search |client |worker )?threads\b|number of (query |search )?threads|batch(ed)? queries|one query at a time",
    "interleave": r"interleav|random(ized|ised)? order|order (of|in which) (the )?(experiments|runs|systems|baselines|evaluations)|alternat(e|ing) (between|runs|the order)|\bRMIT\b|cold.start|warm.up phase",
}
for t in sorted(txt.glob("*.txt")):
    flat = re.sub(r"\s+", " ", t.read_text())
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", flat)
    md = [f"# {t.stem}\n"]
    for var, pat in V.items():
        hits = [s for s in sents if re.search(pat, s, re.I) and len(s) < 700]
        md.append(f"\n## {var} ({len(hits)})")
        seen = set()
        for s in hits:
            key = s[:60]
            if key in seen:
                continue
            seen.add(key)
            md.append(f"- {s[:380]}")
            if len(seen) >= 4:
                break
    (out / f"{t.stem}.md").write_text("\n".join(md))
