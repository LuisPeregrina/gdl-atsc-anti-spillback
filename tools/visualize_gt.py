"""Visualise ground-truth COCO boxes on a random frame of a built MTID view.

Picks a random annotated image from the native-COCO jsons produced by
tools/mtid_coco_build.py and draws its bounding boxes (class colour + label)
on top of the image with OpenCV, then shows it in a window.

Usage (run alongside training):
    python tools/visualize_gt.py --view drone --split val
    python tools/visualize_gt.py --view infra --split train --k 5   # loop 5 random frames
    python tools/visualize_gt.py --view drone --seed 7

Controls while the window is open: 'n' next, 'q'/ESC quit.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

BUILD_ROOT = Path("dataset_build")
VIEW_DIRS = {"drone": "drone", "infra": "infra"}
COLORS = {
    "bicycle": (0, 255, 255),  # BGR
    "car": (255, 0, 0),
    "bus": (0, 255, 0),
    "lorry": (0, 165, 255),
}


def load_json(view: str, split: str) -> dict:
    path = BUILD_ROOT / VIEW_DIRS[view] / "annotations" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def annotated_images(data: dict) -> list[tuple[dict, list[dict]]]:
    """Return [(image, [annotations])] for images that have >= 1 box."""
    ann_by_img: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    out = []
    for img in data["images"]:
        anns = ann_by_img.get(img["id"], [])
        if anns:
            out.append((img, anns))
    return out


def cat_name(data: dict, category_id: int) -> str:
    for c in data["categories"]:
        if c["id"] == category_id:
            return c["name"]
    return str(category_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=list(VIEW_DIRS), default="drone")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--k", type=int, default=1, help="How many frames to show")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None, help="Optional out path (png)")
    args = parser.parse_args()

    view = VIEW_DIRS[args.view]
    data = load_json(view, args.split)
    items = annotated_images(data)
    if not items:
        raise SystemExit(f"No annotated images in {args.view}/{args.split}")

    rng = random.Random(args.seed)
    picks = rng.sample(items, min(args.k, len(items)))
    image_dir = BUILD_ROOT / view / "images"

    shown = 0
    for img, anns in picks:
        image_path = image_dir / img["file_name"]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"WARN: could not read {image_path}")
            continue
        H, W = frame.shape[:2]

        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            name = cat_name(data, ann["category_id"])
            color = COLORS.get(name, (255, 255, 255))
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name}"
            cv2.putText(
                frame, label, (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        title = f"{view}/{args.split} {img['file_name']} ({len(anns)} boxes)"
        print(title)
        if args.save:
            out = Path(args.save) if args.save.endswith(".png") else Path(args.save) / f"{Path(img['file_name']).stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            print(f"  saved {out}")
            continue

        cv2.imshow(title, frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)
        shown += 1
        if key in (ord("q"), 27):
            break

    if not args.save and shown == 0:
        print("Nothing to display.")
    print("done")


if __name__ == "__main__":
    main()
