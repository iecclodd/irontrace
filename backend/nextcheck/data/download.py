"""Bounded download and extraction of UCI dataset 447's raw sensor archive."""

from __future__ import annotations

import hashlib
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
SENSORS = ("TS1", "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "FS1", "FS2", "VS1", "EPS1", "TS2", "TS3", "TS4")
REQUIRED_FILES = {f"{sensor}.txt" for sensor in SENSORS} | {"profile.txt"}
MAX_ARCHIVE_BYTES = 120_000_000
MAX_FILE_BYTES = 125_000_000
MAX_EXTRACTED_BYTES = 850_000_000


class DataValidationError(ValueError):
    """Raw data are absent, malformed, or outside expected bounds."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path, url: str = ARCHIVE_URL) -> dict:
    """Download the official raw archive with a hard byte bound and atomic finish."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size > MAX_ARCHIVE_BYTES:
            raise DataValidationError("Existing archive exceeds size limit")
        return {"url": url, "path": str(destination.resolve()), "sha256": sha256_file(destination), "bytes": destination.stat().st_size}
    part = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "NextCheck/0.1 UCI dataset 447 research preparation"})
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise DataValidationError("Archive download exceeds size limit")
                output.write(chunk)
        if not zipfile.is_zipfile(part):
            raise DataValidationError("Downloaded file is not a ZIP archive")
        part.replace(destination)
    finally:
        part.unlink(missing_ok=True)
    return {"url": url, "path": str(destination.resolve()), "sha256": sha256_file(destination), "bytes": destination.stat().st_size}


def extract_required(archive: Path, raw_dir: Path) -> dict:
    """Extract only the physical sensors and profile; reject unsafe ZIP members."""
    archive, raw_dir = Path(archive), Path(raw_dir)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise DataValidationError("Archive exceeds size limit")
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, zipfile.ZipInfo] = {}
    total = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or info.filename.startswith(("/", "\\")):
                raise DataValidationError(f"Unsafe ZIP path: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise DataValidationError("ZIP symlinks are not permitted")
            if info.is_dir():
                continue
            if info.file_size > MAX_FILE_BYTES:
                raise DataValidationError(f"ZIP member too large: {info.filename}")
            total += info.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise DataValidationError("ZIP extraction exceeds total size limit")
            if name.name in REQUIRED_FILES:
                if name.name in selected:
                    raise DataValidationError(f"Duplicate raw member: {name.name}")
                selected[name.name] = info
        missing = REQUIRED_FILES - selected.keys()
        if missing:
            raise DataValidationError(f"Missing raw files in archive: {sorted(missing)}")
        for basename, info in selected.items():
            target = raw_dir / basename
            temporary = target.with_suffix(".part")
            try:
                with zf.open(info) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise DataValidationError(f"Incomplete extraction: {basename}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return {"raw_dir": str(raw_dir.resolve()), "files": {name: {"bytes": (raw_dir / name).stat().st_size, "sha256": sha256_file(raw_dir / name)} for name in sorted(REQUIRED_FILES)}}
