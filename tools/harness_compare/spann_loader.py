"""indexloader.ini for natively built SPANN indexes, so `indexsearcher` (and ann-suite) can load them.

SPTAG's build tool (`ssdserving`) does not write the top-level indexloader.ini that `indexsearcher`
needs. SPANN's own SaveConfig puts the head (BKT) index's parameters under [BuildHead]; a hand-written
loader that omits them still loads, but the head then uses BKT defaults, i.e. DistCalcMethod=Cosine on an
L2 index: recall drops (0.956 -> 0.897 at 10M) with unchanged IO and QPS. So the head parameters are
copied from head_index/indexloader.ini. VectorPath is blanked: indexsearcher never reads it.
stdlib only (used under the TapeANN venv as well as ann-suite's).
"""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path

K = 10
QUERY_U8 = Path("/home/isfcr/data/bigann/query.u8bin")
# SPTAG (built with BATCH_READ) creates its AIO context once, at index load, sized by the loader's
# SearchInternalResultNum; the command-line value is applied after the load. With 64 here, the kernel's
# rounding up (about 124 events on 24 CPUs) lets IRN <= 128 work by luck, and IRN 256 makes io_submit
# fail ("submitted: Operation not permitted"): postings are skipped and recall collapses silently.
IRN_CAPACITY = 512


def read_ini(path: Path) -> configparser.ConfigParser:
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.optionxform = str  # type: ignore[assignment,method-assign]
    ini.read(path)
    return ini


def write_ini(ini: configparser.ConfigParser, path: Path) -> None:
    with path.open("w") as f:
        ini.write(f, space_around_delimiters=False)


def base_section(root: Path) -> dict[str, str]:
    """[Base] of the build config, with nothing that makes a search load extra data.

    ssdserving loads VectorPath (the full base set) into RAM before searching and again after it for
    recall; indexsearcher never does. Blank it so both load only the index itself.
    """
    base = dict(read_ini(root / "build.ini")["Base"])
    base.update(
        VectorPath="", QueryPath=str(QUERY_U8), TruthPath="", GenerateTruth="false", WarmupPath=""
    )
    return base


def search_params(irn: int, ppl: int) -> dict[str, str]:
    return {
        "ResultNum": str(K),
        "MaxCheck": "4096",
        "SearchInternalResultNum": str(irn),
        "SearchPostingPageLimit": str(ppl),
    }


def write_loader(root: Path, overwrite: bool = False) -> Path:
    """Write <root>/index/indexloader.ini unless one exists (an existing one is left untouched)."""
    index = root / "index"
    target = index / "indexloader.ini"
    if target.exists() and not overwrite:
        return target
    manifest = json.loads((root / "manifest.json").read_text())
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str  # type: ignore[assignment,method-assign]
    base = base_section(root)
    ini["Index"] = {"IndexAlgoType": "SPANN", "ValueType": base["ValueType"]}
    ini["Base"] = base
    ini["SelectHead"] = {"isExecute": "false"}
    head = dict(read_ini(index / "head_index/indexloader.ini")["Index"])
    ini["BuildHead"] = {"isExecute": "false", **head}
    ini["BuildSSDIndex"] = {
        "isExecute": "true",
        "BuildSsdIndex": "false",
        **search_params(IRN_CAPACITY, manifest["posting_page_limit"]),
    }
    write_ini(ini, target)
    return target


def ensure_irn_capacity(root: Path, capacity: int = IRN_CAPACITY) -> bool:
    """Raise an existing loader's SearchInternalResultNum to `capacity` (original kept as .bak).

    Only that one line changes. It is the AIO context size at load, so any per-run IRN up to
    `capacity` is then honoured; it does not set the IRN used for a search (the command line does).
    """
    target = root / "index" / "indexloader.ini"
    text = target.read_text()
    match = re.search(r"^SearchInternalResultNum=(\d+)$", text, flags=re.MULTILINE)
    if match is None or int(match.group(1)) >= capacity:
        return False
    backup = target.with_name(target.name + ".bak")
    if not backup.exists():
        backup.write_text(text)
    target.write_text(
        text[: match.start()] + f"SearchInternalResultNum={capacity}" + text[match.end() :]
    )
    return True
