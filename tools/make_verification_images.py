"""Make geometry variants of a real input image for deployment verification."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="data/verification_images")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error("input image is unreadable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "portrait.jpg"), image)
    landscape = np.full((721, 1281, 3), 114, np.uint8)
    scale = min(720 / image.shape[0], 1280 / image.shape[1])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    left = (landscape.shape[1] - resized.shape[1]) // 2
    landscape[:resized.shape[0], left:left + resized.shape[1]] = resized
    cv2.imwrite(str(output / "landscape-odd.jpg"), landscape)
    cv2.imwrite(str(output / "small.jpg"), cv2.resize(image, (203, 271)))
    print(f"Created 3 image variants in {output}; these are conversion checks, not an independent accuracy dataset.")


if __name__ == "__main__":
    main()
