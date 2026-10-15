"""SSD state probe: random 4 KB O_DIRECT reads at queue depth 1 from one file.

On isfcr's QLC SSD a freshly written file sits in the SLC cache and reads ~3x
faster (~31 us vs ~92 us) than one folded to QLC; timing a search on such a file
flatters it. Run on every index file before an end-to-end comparison.

  python tools/evolve/e2e/ssd_qd1.py FILE [FILE ...]
"""

import mmap
import os
import random
import sys
import time

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    n = os.fstat(fd).st_size // 4096
    buf = mmap.mmap(-1, 4096)
    random.seed(1)
    lat = []
    for _ in range(4000):
        off = random.randrange(n) * 4096
        t = time.perf_counter()
        os.preadv(fd, [buf], off)
        lat.append((time.perf_counter() - t) * 1e6)
    os.close(fd)
    lat.sort()
    print(f"qd1 p50 {lat[2000]:.0f} us  p90 {lat[3600]:.0f} us  {path}", flush=True)
