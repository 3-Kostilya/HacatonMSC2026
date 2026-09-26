"""Stable fingerprints of frozen Git text across LF/CRLF checkouts."""

import hashlib
from pathlib import Path


def frozen_rule_sha256(path: Path) -> str:
    """The R6 freeze was pinned using LF; ignore only Git's CRLF conversion.

    Whitespace, field order, values, encoding and trailing-newline changes still
    change this fingerprint. This is not semantic JSON reserialization.
    """
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in raw:
        raise ValueError("unsupported carriage return in frozen rule")
    return hashlib.sha256(raw).hexdigest()
