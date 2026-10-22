"""Symlinks that give ann-suite one entry point to data and indices stored elsewhere.

A manifest (``data/links.yaml``) maps a link directory inside the repo to a list of
absolute targets; a target may be a glob. Each target becomes a symlink named after its
basename, e.g. ``indices/starling: [/home/isfcr/starling/indices/*]`` creates
``indices/starling/sift1b -> /home/isfcr/starling/indices/sift1b``.

Links are absolute, so the container runner can bind-mount each target at the same path
(see ``symlink_target_mounts``) and the links resolve inside containers too.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/links.yaml")


@dataclass
class LinkReport:
    """Outcome of applying or checking a manifest."""

    ok: list[Path] = field(default_factory=list)
    created: list[Path] = field(default_factory=list)
    missing: list[Path] = field(default_factory=list)  # link absent (check mode)
    wrong: list[tuple[Path, Path]] = field(default_factory=list)  # (link, current target)
    blocked: list[Path] = field(default_factory=list)  # a real file/dir sits at the link path
    unmatched: list[str] = field(default_factory=list)  # target or glob with no match
    dangling: list[Path] = field(default_factory=list)  # broken links found under link dirs

    @property
    def problems(self) -> int:
        return (
            len(self.missing)
            + len(self.wrong)
            + len(self.blocked)
            + len(self.unmatched)
            + len(self.dangling)
        )


def load_manifest(path: Path) -> dict[str, list[str]]:
    """Load ``{link_dir: [absolute target or glob, ...]}`` from YAML."""
    data = yaml.safe_load(path.read_text()) or {}
    links = data.get("links", {})
    if not isinstance(links, dict):
        raise ValueError(f"{path}: 'links' must map link directories to target lists")
    for link_dir, targets in links.items():
        if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
            raise ValueError(f"{path}: links.{link_dir} must be a list of paths")
        for t in targets:
            if not Path(t).is_absolute():
                raise ValueError(f"{path}: target '{t}' must be absolute")
    return links


def expand_links(links: dict[str, list[str]], root: Path) -> tuple[dict[Path, Path], list[str]]:
    """Resolve the manifest into ``{link_path: target}`` plus targets that matched nothing."""
    wanted: dict[Path, Path] = {}
    unmatched: list[str] = []
    for link_dir, targets in links.items():
        for pattern in targets:
            matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
            matches = [m for m in matches if Path(m).exists()]
            if not matches:
                unmatched.append(pattern)
            for m in matches:
                target = Path(m)
                link = root / link_dir / target.name
                if link in wanted and wanted[link] != target:
                    raise ValueError(f"two targets map to {link}: {wanted[link]} and {target}")
                wanted[link] = target
    return wanted, unmatched


def apply_links(manifest: Path, root: Path, check_only: bool = False) -> LinkReport:
    """Create (or, with ``check_only``, verify) every link in the manifest.

    Never deletes or overwrites a real file or directory. A link that points elsewhere is
    reported, not changed.
    """
    links = load_manifest(manifest)
    wanted, unmatched = expand_links(links, root)
    report = LinkReport(unmatched=unmatched)
    for link, target in sorted(wanted.items()):
        if link.is_symlink():
            current = Path(link.readlink())
            if current == target:
                report.ok.append(link)
            else:
                report.wrong.append((link, current))
        elif link.exists():
            report.blocked.append(link)
        elif check_only:
            report.missing.append(link)
        else:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
            report.created.append(link)
    for link_dir in links:
        base = root / link_dir
        if base.is_dir():
            report.dangling += sorted(
                p for p in base.iterdir() if p.is_symlink() and not p.exists()
            )
    return report


def symlink_target_mounts(roots: list[Path], depth: int = 2) -> list[Path]:
    """Real targets of absolute symlinks under ``roots`` that lie outside every root.

    The runner binds each one at its own path so absolute links resolve in containers.
    Nested targets are dropped in favour of their parent. Relative links that escape a
    root cannot resolve in a container and are only logged.
    """
    roots = [r.resolve() for r in roots]
    found: set[Path] = set()

    def inside_roots(p: Path) -> bool:
        return any(p == r or p.is_relative_to(r) for r in roots)

    def walk(d: Path, level: int) -> None:
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        for p in entries:
            if p.is_symlink():
                if not p.exists():
                    continue
                real = p.resolve()
                if inside_roots(real):
                    continue
                if Path(p.readlink()).is_absolute():
                    found.add(real)
                else:
                    logger.warning(f"Relative link {p} escapes the mounted dirs; it will dangle")
            elif level < depth and p.is_dir():
                walk(p, level + 1)

    for r in roots:
        if r.is_dir():
            walk(r, 1)
    ordered = sorted(found)
    return [p for p in ordered if not any(p != q and p.is_relative_to(q) for q in ordered)]
