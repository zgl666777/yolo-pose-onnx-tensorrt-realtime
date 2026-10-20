"""Generate explicitly synthetic records to exercise classifier tooling."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from poseguard.demo import synthetic_person


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic-poses.jsonl")
    parser.add_argument("--subjects", type=int, default=10)
    args = parser.parse_args()
    if args.subjects < 5:
        parser.error("at least 5 subjects are required")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for person in range(args.subjects):
            for label, angle in (("standing", 0), ("sitting", 0.1),
                                 ("lying", math.pi / 2), ("falling", math.pi / 3)):
                for index in range(20):
                    det = synthetic_person((320, 280), angle + rng.normal(0, 0.02), scale=110)
                    if label == "sitting":
                        det.keypoints[13:, 1] -= 65
                        det.bbox[3] -= 65
                    det.keypoints[:, :2] += rng.normal(0, 1, (17, 2))
                    row = dict(video_id=f"synthetic-video-{person}", person_id=f"synthetic-person-{person}",
                               timestamp=count / 30, bbox=det.bbox.tolist(), keypoints=det.keypoints.tolist(),
                               label=label, synthetic=True)
                    file.write(json.dumps(row) + "\n")
                    count += 1
    print(json.dumps(dict(output=str(path), records=count, synthetic=True)))


if __name__ == "__main__":
    main()

