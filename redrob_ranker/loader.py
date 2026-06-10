"""Streaming reader for the candidate pool.

The full pool (`candidates.jsonl`) is ~487 MB / 100k records, so we stream line by
line rather than loading it all as one blob. JSON parsing of 487 MB is a real chunk
of the 5-minute budget, so we use ``orjson`` when available (~5-10x faster) and fall
back to the stdlib ``json`` module otherwise.

Supports plain `.jsonl` and gzipped `.jsonl.gz` transparently (the validator and
README both reference the gzipped form).
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterator, Optional, Union

from .schema import Candidate

try:  # optional fast path
    import orjson

    def _loads(line: bytes):
        return orjson.loads(line)

    _BINARY = True
except ImportError:  # pragma: no cover - environment dependent
    def _loads(line):
        return json.loads(line)

    _BINARY = False


PathLike = Union[str, Path]


def _open_text(path: Path):
    """Open .jsonl or .jsonl.gz, in binary if orjson is available else text."""
    is_gz = path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
    mode_bin = "rb"
    mode_txt = "rt"
    if is_gz:
        return gzip.open(path, mode_bin if _BINARY else mode_txt,
                         encoding=None if _BINARY else "utf-8")
    if _BINARY:
        return open(path, "rb")
    return open(path, "rt", encoding="utf-8")


def iter_raw(path: PathLike, limit: Optional[int] = None) -> Iterator[dict]:
    """Yield raw candidate dicts, streaming. Skips blank lines; tolerates bad lines."""
    path = Path(path)
    n = 0
    with _open_text(path) as fh:
        for line in fh:
            if not line or (isinstance(line, (bytes, bytearray)) and not line.strip()):
                continue
            if isinstance(line, str) and not line.strip():
                continue
            try:
                obj = _loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                yield obj
                n += 1
                if limit is not None and n >= limit:
                    return


def iter_candidates(path: PathLike, limit: Optional[int] = None) -> Iterator[Candidate]:
    """Yield parsed :class:`Candidate` objects, streaming."""
    for d in iter_raw(path, limit=limit):
        yield Candidate.from_dict(d)


def load_candidates(path: PathLike, limit: Optional[int] = None) -> list[Candidate]:
    """Eagerly load the pool into a list (fine for 100k on 16 GB)."""
    return list(iter_candidates(path, limit=limit))


def load_sample_json(path: PathLike) -> list[Candidate]:
    """Load the pretty-printed `sample_candidates.json` (a JSON array, not JSONL)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Candidate.from_dict(d) for d in data if isinstance(d, dict)]


def count_lines(path: PathLike) -> int:
    path = Path(path)
    n = 0
    with _open_text(path) as fh:
        for line in fh:
            if isinstance(line, (bytes, bytearray)):
                if line.strip():
                    n += 1
            elif line.strip():
                n += 1
    return n
