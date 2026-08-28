"""HTTP dataset download with atomic writes and Rich progress.

Downloads a URL to a destination file. The file is written to a temporary path
in the same directory and atomically renamed into place only on success, so a
failed or interrupted download never leaves a truncated file that would later
masquerade as "already downloaded".
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import Request, urlopen

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger(__name__)

#: Custom User-Agent to avoid 403 Forbidden responses from CDNs.
USER_AGENT = "Mozilla/5.0 (compatible; ann-suite/0.1.0; +https://github.com/ann-suite)"

_BLOCK_SIZE = 64 * 1024


def _progress_bar() -> Progress:
    """Build a Rich progress bar configured for file downloads."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )


def download_file(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> None:
    """Download ``url`` to ``dest``.

    Args:
        url: Source URL.
        dest: Destination file path.
        force: Re-download even if ``dest`` already exists.
        show_progress: Render a Rich progress bar (disabled under ``--quiet``).
    """
    dest = dest.resolve()

    if dest.exists() and not force:
        logger.info("Already exists, skipping: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)

    request = Request(url, headers={"User-Agent": USER_AGENT})

    # Download to a temp file in the same directory, then rename atomically.
    tmp_path = dest.with_suffix(f"{dest.suffix}.part")
    with urlopen(request) as response, tmp_path.open("wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))

        if show_progress:
            with _progress_bar() as progress:
                task = progress.add_task(f"[cyan]{dest.name}", total=total or None)
                while True:
                    chunk = response.read(_BLOCK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress.advance(task, len(chunk))
        else:
            while True:
                chunk = response.read(_BLOCK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)

    tmp_path.replace(dest)
    logger.info("Downloaded %d bytes to %s", dest.stat().st_size, dest)
