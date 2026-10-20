"""Download selected original URFD RGB archives for diagnostic review."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.request import urlopen, Request
import zipfile

from poseguard.annotations import file_sha256, write_jsonl
from poseguard.image_io import read_image


BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
SEQUENCES = {f"{kind}-{index:02d}" for kind, count in (("fall", 30), ("adl", 40))
             for index in range(1, count + 1)}


def safe_member_path(destination, member):
    name = member.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or ":" in name or not path.parts or
            any(part in {"", "."} for part in path.parts)):
        raise ValueError(f"Unsafe archive path: {member.filename}")
    if (member.external_attr >> 16) & 0o170000 == 0o120000:
        raise ValueError(f"Archive symlink is not permitted: {member.filename}")
    target = (Path(destination).resolve() / Path(*path.parts)).resolve()
    if not target.is_relative_to(Path(destination).resolve()):
        raise ValueError(f"Archive path escapes extraction directory: {member.filename}")
    return target


def download_archive(url, archive, parallel=1, max_seconds=480, resume_parts=False):
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Download time limit must be finite and positive")
    if not isinstance(parallel, int) or isinstance(parallel, bool) or not 1 <= parallel <= 8:
        raise ValueError("Parallel download count must be an integer from 1 to 8")
    started = time.monotonic()
    with urlopen(Request(url, method="HEAD"), timeout=30) as response:
        content_length = int(response.headers["Content-Length"])
    if content_length <= 0 or content_length > 128 * 1024 ** 2:
        raise ValueError("RGB archive exceeds the diagnostic 128 MiB limit")
    def remaining_timeout():
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Original RGB download exceeded total time limit")
        return min(30, remaining)
    if parallel == 1:
        with urlopen(url, timeout=remaining_timeout()) as response, archive.open("wb") as stream:
            total = 0
            for chunk in iter(lambda: response.read(256 * 1024), b""):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Original RGB download exceeded total time limit")
                total += len(chunk)
                if total > content_length or total > 128 * 1024 ** 2:
                    raise ValueError("Original RGB response exceeds declared or permitted archive size")
                stream.write(chunk)
    else:
        parts = archive.parent / "download-parts"
        parts.mkdir(exist_ok=resume_parts)
        ranges = [(start, min(start + 1024 ** 2 - 1, content_length - 1))
                  for start in range(0, content_length, 1024 ** 2)]
        def fetch(part):
            index, (start, end) = part
            path = parts / f"part-{index:04d}.bin"
            if resume_parts and path.is_file() and path.stat().st_size == end - start + 1:
                return path
            existing = path.stat().st_size if resume_parts and path.is_file() else 0
            if existing > end - start + 1:
                raise ValueError("Existing byte part exceeds requested span")
            request_start = start + existing
            with urlopen(Request(url, headers={"Range": f"bytes={request_start}-{end}"}), timeout=remaining_timeout()) as response, path.open("ab" if existing else "wb") as stream:
                if response.status != 206:
                    raise ValueError("Server did not honor byte Range request")
                expected_range = f"bytes {request_start}-{end}/{content_length}"
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError(f"Unexpected Content-Range; expected {expected_range}")
                if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) != end - request_start + 1:
                    raise ValueError("Range Content-Length differs from requested byte span")
                length = existing
                for chunk in iter(lambda: response.read(64 * 1024), b""):
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError("Original RGB ranged download exceeded total time limit")
                    length += len(chunk)
                    if length > end - start + 1:
                        raise ValueError("Original RGB range exceeds requested span")
                    stream.write(chunk)
            if length != end - start + 1:
                raise ValueError("Incomplete original RGB range download")
            print(f'{archive.name}: part {index + 1}/{len(ranges)} verified', flush=True)
            return path
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            paths = list(pool.map(fetch, enumerate(ranges)))
        with archive.open("wb") as stream:
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, stream)
        total = archive.stat().st_size
    if total != content_length:
        raise ValueError(f"Incomplete download: {total}/{content_length} bytes")
    return total, content_length


def download_sequence(sequence, output, sync_root, parallel=1, max_seconds=480, resume_parts=False):
    if sequence not in SEQUENCES:
        raise ValueError("Sequence must be one of the fixed URFD fall-01..30 or adl-01..40 names")
    root = Path(output) / sequence
    if root.exists() and any(root.iterdir()) and not (resume_parts and (root / "download-parts").is_dir()):
        raise FileExistsError(f"Use a new directory for {sequence}")
    root.mkdir(parents=True, exist_ok=resume_parts)
    archive = root / f"{sequence}-cam0-rgb.zip"
    url = BASE + archive.name
    total, content_length = download_archive(url, archive, parallel, max_seconds, resume_parts)
    extracted = root / "frames"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if sum(entry.file_size for entry in entries) > 1024 ** 3:
            raise ValueError("RGB archive expands beyond 1 GiB limit")
        safe_paths = [(entry, safe_member_path(extracted, entry)) for entry in entries]
        corrupt = package.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC verification failed: {corrupt}")
        for entry, destination in safe_paths:
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.suffix.lower() != ".png":
                raise ValueError(f"Unexpected non-PNG original RGB member: {entry.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with package.open(entry) as source, destination.open("wb") as stream:
                shutil.copyfileobj(source, stream)
    timestamp_path = Path(sync_root) / f"{sequence}-data.csv"
    with timestamp_path.open(newline="", encoding="utf-8") as stream:
        times = {int(row[0]): float(row[1]) / 1000 for row in csv.reader(stream)}
    images, tasks = [], []
    seen = set()
    for path in sorted(extracted.rglob("*.png")):
        matches = re.findall(r"\d+", path.stem)
        if not matches:
            raise ValueError(f"Original RGB filename has no frame number: {path.name}")
        source_frame = int(matches[-1])
        if source_frame in seen:
            raise ValueError(f"Duplicate original RGB source frame: {source_frame}")
        seen.add(source_frame)
        image = read_image(path)
        if image is None:
            raise ValueError(f"Unreadable original RGB PNG: {path}")
        h, w = image.shape[:2]
        timestamp = times.get(source_frame)
        if timestamp is None:
            raise ValueError(f"No official source timestamp for original RGB frame {source_frame}")
        images.append(dict(path=str(path.relative_to(root)), source_frame=source_frame,
                           width=w, height=h, bytes=path.stat().st_size, sha256=file_sha256(path)))
        tasks.append(dict(image=str(path.resolve()), video_id=f"{sequence}-cam0", frame_id=source_frame - 1,
                          source_frame=source_frame, timestamp=timestamp, source_timestamp=timestamp,
                          source_roi=None, media_scope="original_full_resolution_RGB",
                          bbox=None, person_id="", label=None, identity_verified=False,
                          annotation_source="draft", review_status="pending", authoritative=False,
                          human_review_pending=True, review_notes="Original RGB diagnostic review; verify all people and events manually"))
    images.sort(key=lambda r: r["source_frame"])
    tasks.sort(key=lambda r: r["source_frame"])
    if not tasks:
        raise ValueError("Archive has no RGB images")
    tasks_path = root / "review-tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "sequence": sequence,
                "scope": "original_RGB_diagnostic_review_not_manual_truth", "authoritative": False,
                "human_review_pending": True, "license": "CC BY-NC-SA 4.0; non-commercial academic use",
                "source_url": url, "archive_bytes": total, "declared_content_length": content_length,
                "archive_sha256": file_sha256(archive), "zip_crc_verified": True,
                "download_strategy": "parallel_byte_ranges" if parallel > 1 else "single_connection",
                "resumed_size_checked_parts": resume_parts,
                "range_validation_scope": "newly fetched ranges validate Content-Range; resumed parts validate byte span and whole ZIP CRC",
                "archive_paths_verified": True, "frames": len(tasks),
                "image_dimensions": sorted({(r["width"], r["height"]) for r in images}),
                "timestamps_path": str(timestamp_path.resolve()), "timestamps_sha256": file_sha256(timestamp_path),
                "tasks_sha256": file_sha256(tasks_path), "images": images,
                "limitations": ["No human identities, boxes, posture labels or reviewed fall intervals were assigned",
                                "Original RGB and depth sync require human alignment review"]}
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=sorted(SEQUENCES), action="append", required=True)
    parser.add_argument("--output", default="data/urfall-original-rgb")
    parser.add_argument("--sync-data", default="data/urfall")
    parser.add_argument("--parallel", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--max-download-seconds", type=float, default=480)
    parser.add_argument("--resume-parts", action="store_true", help="Reuse correctly-sized byte parts; whole ZIP CRC still required")
    args = parser.parse_args()
    for sequence in args.sequence:
        try:
            report = download_sequence(sequence, args.output, args.sync_data, args.parallel, args.max_download_seconds,
                                       args.resume_parts)
        except Exception as exc:
            destination = Path(args.output) / sequence
            if destination.is_dir():
                (destination / "download-failure.json").write_text(json.dumps({"status": "failed",
                    "sequence": sequence, "scope": "partial_download_not_verified_not_usable",
                    "error": f"{type(exc).__name__}: {exc}", "authoritative": False}, indent=2), encoding="utf-8")
            raise
        print(json.dumps({key: report[key] for key in ("sequence", "archive_bytes", "archive_sha256", "frames",
                                                      "image_dimensions", "zip_crc_verified", "archive_paths_verified")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
