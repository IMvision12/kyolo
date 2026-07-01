"""Download and cache remote files (backed by ``keras.utils.get_file``).

Mirrors the KerasFormers downloader: given a URL, fetch it once into a local
cache directory and return the path. Used to pull user-supplied checkpoint URLs
during weight conversion.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from keras import utils

__all__ = ["validate_url", "download_file", "DEFAULT_CACHE"]

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kyolo")


def validate_url(url: str) -> bool:
    """Return True if ``url`` is a well-formed http(s) URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_file(file_url: str, cache_dir=None, force_download: bool = False) -> str:
    """Download ``file_url`` into ``cache_dir`` and return the local path.

    Args:
        file_url: http(s) URL to fetch.
        cache_dir: destination directory (default ``~/.cache/kyolo``).
        force_download: re-download even if a cached copy exists.

    Returns:
        Absolute path to the downloaded file.
    """
    if not file_url:
        raise ValueError("file_url cannot be empty")
    if not validate_url(file_url):
        raise ValueError(f"Invalid URL format: {file_url}")

    cache_dir = Path(cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_name = os.path.basename(urlparse(file_url).path) or "download"
    local_file = cache_dir / file_name
    if local_file.exists() and not force_download:
        print(f"Found cached file at {local_file}")
        return str(local_file)

    return utils.get_file(
        fname=file_name,
        origin=file_url,
        cache_dir=str(cache_dir),
        cache_subdir="",
        extract=False,
        force_download=force_download,
    )
