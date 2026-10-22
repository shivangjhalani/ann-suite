"""Shared constants for the ANN Suite.

Centralized constants to avoid duplication and ensure consistency across modules.
"""

from __future__ import annotations

# Standard page size in bytes (4KB)
# Used for cross-system comparable page metrics regardless of physical block size.
STANDARD_PAGE_SIZE = 4096

# Maximum number of log files to keep per algorithm/mode combination.
# Older log files beyond this limit are automatically cleaned up.
MAX_LOG_FILES_PER_TYPE = 50

# Environment variable holding the sudo password for passworded sudo (page-cache drops,
# NVMe SMART reads). Never logged or persisted.
SUDO_PASSWORD_ENV = "ANN_SUITE_SUDO_PASSWORD"
