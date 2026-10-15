"""Helpers for run_heldout_e2e.sh.

heldout_paths.py index DATASET STAGE DATASET_NAME [PROGRAM]
                                                   cached index dir of PROGRAM (default
                                                   abl_fastscan_g40_DATASET.py)
heldout_paths.py grid NPROBE:SHORTLIST,... PAGES,...  JSON list of search points (the product)
heldout_paths.py points DATASET                    its SEARCH_POINTS as JSON
heldout_paths.py flush FILE GB                     write+fsync+delete GB of random data, to
                                                   fold recent writes out of the SSD's SLC cache
"""

import ast
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def program(ds: str) -> str:
    return (HERE.parent / "ablation" / f"abl_fastscan_g40_{ds}.py").read_text()


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "index":
        import evolve_bench as eb

        ds, stage, name = sys.argv[2:5]
        src = Path(sys.argv[5]).read_text() if len(sys.argv) > 5 else program(ds)
        cfg = eb._cfg(eb.REPO / "configs/evolve/bigann10m.yaml")
        print(Path(cfg["index_dir"]) / "cache" / stage / eb._build_hash(src, stage, name))
    elif cmd == "grid":
        pairs = [tuple(map(int, x.split(":"))) for x in sys.argv[2].split(",")]
        pages = [int(x) for x in sys.argv[3].split(",")]
        pts = [{"nprobe": a, "shortlist": b, "pages": c} for a, b in pairs for c in pages]
        print(json.dumps(pts))
    elif cmd == "points":
        tree = ast.parse(program(sys.argv[2]))
        for n in tree.body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SEARCH_POINTS":
                print(json.dumps(ast.literal_eval(n.value)))
    elif cmd == "flush":
        path, gb = sys.argv[2], int(sys.argv[3])
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        blk = os.urandom(64 << 20)
        for i in range(gb * 16):
            os.write(fd, blk[i % 7 :] + blk[: i % 7])
        os.fsync(fd)
        os.close(fd)
        os.unlink(path)


if __name__ == "__main__":
    main()
