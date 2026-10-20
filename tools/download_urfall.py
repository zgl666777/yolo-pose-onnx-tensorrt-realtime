"""Download original URFD RGB videos and source labels for academic evaluation."""
from __future__ import annotations
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import cv2

PAGE = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/urfall")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    names = [f"{kind}-{i:02d}{suffix}" for kind, count in (("fall", 30), ("adl", 40))
             for i in range(1, count + 1) for suffix in ("-cam0.mp4", "-data.csv")]
    names += ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
    def fetch(name):
        dest = root / name
        try:
            if not dest.exists():
                with urlopen(BASE + name, timeout=30) as response:
                    data = response.read()
                temporary = dest.with_suffix(dest.suffix + ".partial")
                temporary.write_bytes(data)
                temporary.replace(dest)
            data = dest.read_bytes()
            row = dict(file=name, url=BASE + name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if name.endswith(".mp4"):
                cap = cv2.VideoCapture(str(dest))
                row.update(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                           width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cap.release()
                if row["frames"] == 0 or row["fps"] <= 0:
                    raise ValueError("Downloaded file is not a readable video")
            return row
        except Exception as exc:
            return dict(file=name, url=BASE + name, error=str(exc))
    with urlopen(PAGE, timeout=30) as response:
        (root / "official-page.html").write_bytes(response.read())
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, names):
            rows.append(row)
            print(f"{len(rows)}/{len(names)} {row['file']} {'FAILED' if 'error' in row else 'OK'}", flush=True)
    manifest = dict(dataset="UR Fall Detection Dataset", source=PAGE,
                    downloaded_utc=datetime.now(timezone.utc).isoformat(), license="CC BY-NC-SA 4.0; non-commercial academic use",
                    citation="Kwolek & Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, CMPB 117(3), 2014, 489-501",
                    labels="-1=not lying (standing/sitting unresolved), 0=temporary transition (not accidental-fall truth), 1=lying on ground; depth frame labels",
                    limitations=["Cross-video participant identities are not supplied on the official page",
                                 "Manual RGB boxes and standing/sitting labels still required",
                                 "Same participant may occur in many videos; no person-independent claim yet"],
                    files=rows)
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Unknown IDs intentionally remain blank: per-video IDs would falsely imply independence.
    import csv
    registry = root / "video-registry.csv"
    if not registry.exists():
        with registry.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "path", "person_id", "identity_verified", "split", "annotation_status"])
            writer.writeheader()
            for row in rows:
                if row["file"].endswith(".mp4") and "error" not in row:
                    writer.writerow(dict(video_id=row["file"][:-4], path=row["file"], person_id="",
                                         identity_verified="false", split="unassigned", annotation_status="pending"))
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
