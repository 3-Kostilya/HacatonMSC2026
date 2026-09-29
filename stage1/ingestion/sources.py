"""Bounded-memory readers for event CSV files and 7-Zip containers."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from typing import BinaryIO

from stage1.normalization import EVENT_FIELDS


def find_seven_zip() -> str:
    configured = os.environ.get("SEVEN_ZIP")
    candidates = [configured] if configured else ["7zz", "7z", r"C:/Program Files/7-Zip/7z.exe"]
    for candidate in candidates:
        if candidate and (resolved := shutil.which(candidate)):
            return resolved
    raise FileNotFoundError("7-Zip not found; install 7z/7zz or set SEVEN_ZIP")


def parse_7z_listing(text: str) -> list[dict[str, str]]:
    """Return file-member records from a ``7z l -slt`` listing."""
    tail = text.split("----------", 1)
    if len(tail) != 2:
        raise ValueError("Unexpected 7-Zip listing: member delimiter absent")
    records = []
    for block in re.split(r"\r?\n\r?\n", tail[1].strip()):
        record = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                record[key] = value
        if record.get("Path"):
            records.append(record)
    return records


def _archive_members(text: str) -> list[str]:
    """Extract non-directory member names from ``7z l -slt`` output."""

    members = []
    for record in parse_7z_listing(text):
        name = record.get("Path")
        if name and not record.get("Attributes", "").startswith("D"):
            members.append(name)
    return members


def _validate_header(header: list[str], source: str) -> None:
    if header != list(EVENT_FIELDS):
        raise ValueError(
            f"{source}: invalid header at source row 1; expected {list(EVENT_FIELDS)!r}, got {header!r}"
        )


def _iter_csv(binary: BinaryIO, source: str, max_rows: int | None) -> Iterator[dict[str, str]]:
    # utf-8-sig accepts ordinary UTF-8 and removes a leading BOM only.
    import io

    # Normalize physical CRLF record endings while retaining embedded LF text.
    text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="\n")
    reader = csv.reader(text, strict=True)
    try:
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{source}: missing header at source row 1") from None
        except csv.Error as error:
            raise ValueError(f"{source}: malformed CSV at source row 1: {error}") from error
        _validate_header(header, source)

        emitted = 0
        for record_number, values in enumerate(reader, start=2):
            if len(values) != len(EVENT_FIELDS):
                raise ValueError(
                    f"{source}: expected {len(EVENT_FIELDS)} fields at source row {record_number}, "
                    f"got {len(values)}"
                )
            yield {
                **dict(zip(EVENT_FIELDS, values)),
                "__source__": source,
                "__source_row__": record_number,
            }
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return emitted
        return emitted
    except csv.Error as error:
        # csv.Error does not always contain a useful physical line; record_num
        # still identifies the source record that was being consumed.
        raise ValueError(
            f"{source}: malformed CSV at source row {reader.line_num}: {error}"
        ) from error
    finally:
        text.detach()


def _iter_archive(path: Path, source: str, max_rows: int | None) -> Iterator[dict[str, str]]:
    executable = find_seven_zip()
    listing = subprocess.run(
        [executable, "l", "-slt", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if listing.returncode:
        detail = listing.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{source}: unable to list 7z archive (exit {listing.returncode}): {detail}"
        )
    members = _archive_members(listing.stdout.decode("utf-8", errors="replace"))
    if len(members) != 1 or not members[0].casefold().endswith(".csv"):
        raise ValueError(f"{source}: archive must contain exactly one CSV file; found {members!r}")

    stderr = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [executable, "e", "-so", "--", str(path), members[0]],
        stdout=subprocess.PIPE,
        stderr=stderr,
    )
    assert process.stdout is not None
    try:
        emitted = yield from _iter_csv(process.stdout, source, max_rows)
        intentional_stop = max_rows is not None and emitted == max_rows
        if intentional_stop and process.poll() is None:
            # A bounded read intentionally leaves unread archive output.  Stop
            # the extractor and do not treat its resulting exit as a failure.
            process.terminate()
            process.wait(timeout=10)
        else:
            return_code = process.wait()
            if return_code:
                stderr.seek(0)
                error = stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"{source}: 7z extraction failed (exit {return_code}): {error}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()
        stderr.close()


def iter_source_rows(path: Path, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    """Yield lossless event rows from a CSV or a single-member 7z archive."""

    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0
    ):
        raise ValueError("max_rows must be None or a positive integer")
    source_path = Path(path).resolve(strict=False)
    source = str(source_path)

    def generate() -> Iterator[dict[str, str]]:
        if source_path.suffix.casefold() == ".7z":
            yield from _iter_archive(source_path, source, max_rows)
        else:
            with source_path.open("rb") as stream:
                yield from _iter_csv(stream, source, max_rows)

    return generate()
