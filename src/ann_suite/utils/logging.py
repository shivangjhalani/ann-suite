"""Logging configuration for ANN Suite."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rich_console: bool = True,
    json_format: bool = False,
) -> None:
    """Configure logging for the benchmark suite.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        rich_console: Use rich console handler for pretty output
        json_format: Use structured JSON logging format (overrides rich_console)
    """
    handlers: list[logging.Handler] = []

    if json_format:
        # Structured JSON logging for programmatic parsing
        import json
        from datetime import datetime

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_obj = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_obj)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handlers.append(handler)
    elif rich_console:
        try:
            from rich.logging import RichHandler

            handlers.append(
                RichHandler(
                    level=level,
                    rich_tracebacks=True,
                    markup=True,
                    show_time=True,
                    show_path=False,
                )
            )
        except ImportError:
            # Fall back to standard handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handlers.append(handler)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
