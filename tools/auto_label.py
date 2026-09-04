"""Stage 2: pseudo-label the unannotated MTID frames with a trained LibreYOLO model.

Given a trained per-view baseline (e.g. the Colab-trained ``LibreYOLO9c`` for
``Drone``) and the Stage-0 build tree produced by ``tools/mtid_coco_build.py``,
this labels a sampled subset of the still-unannotated ``frames/`` pool:

  1. The annotated set (ground truth) is the union of the Stage-0 train/val
     jsons. Every other frame in the view's ``frames/`` folder is "unannotated".
  2. Pool = unannotated frames, sampled every ``--stride``-th one (default 30,
     ~1 frame/sec at 30 fps). Deterministic so reruns are reproducible.
  3. The baseline runs inference on each pool frame; boxes below ``--conf`` are
     dropped. Frames with no surviving box are skipped (they add nothing).
  4. Accepted boxes are written as COCO annotations (category id = label + 1,
     matching Stage-0 [bicycle,car,bus,lorry] = ids 1..4) into a *candidate*
     json ``annotations/pseudo_<view>.json``. The labeled frames are symlinked
     into the same ``images/`` folder so LibreYOLO's native-COCO loader can read
     them.
  5. A merged training json ``annotations/train_pseudo_<view>.json`` is written =
     Stage-0 GT *train* set + accepted pseudo-labels. Point ``annotations.train``
     at it to retrain on the expanded set.

The candidate json is intentionally separate (review it, tweak the conf
threshold, re-run) before you commit to the merged training json. Ids are kept
unique across the merge so GT and pseudo annotations never collide.

Immutable-source guarantee: never writes to the kagglehub cache; it only reads
source frames and symlinks/copies them into the build tree.

Run where the trained weights live (a Colab GPU is fastest). Example:

    python -m tools.auto_label \\
        --view Drone \\
        --build dataset_build/drone \\
        --source /content/.../versions/1 \\
        --weights /content/drive/.../baseline_runs/mtid_drone/weights/best.pt \\
        --stride 30 --conf 0.60
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2

from .mtid_coco_build import (
    CocoAnnotation,
    CocoCategory,
    CocoImage,
    CATEGORY_NAMES,
)

FRAME_RE = re.compile(r"_(\d+)\.jpg$")


# --------------------------------------------------------------------------- #
# Frame helpers
# --------------------------------------------------------------------------- #


def frame_number(file_name: str) -> int:
    m = FRAME_RE.search(file_name)
    if m is None:
        raise ValueError(f"cannot parse frame number from {file_name!r}")
    return int(m.group(1))


def annotated_basenames(build_dir: Path) -> set[str]:
    """Union of file_names across the Stage-0 train/val jsons of a view."""
    names: set[str] = set()
    for split in ("train", "val"):
        path = build_dir / "annotations" / f"{split}.json"
        if path.exists():
            data = json.loads(path.read_text())
            names.update(img["file_name"] for img in data["images"])
    return names


def list_pool_frames(
    source_root: Path,
    view_key: str,
    annotated: set[str],
    *,
    stride: int = 30,
    offset: int = 0,
) -> list[Path]:
    """Unannotated ``frames/`` files of a view, sampled every ``stride``-th.

    ``frames/`` holds one file per frame number; the annotated set is the union
    of ground-truth jsons. Sort numerically (frame number order) for determinism,
    drop annotated frames, then keep indices ``offset, offset+stride, ...``.
    """
    from .mtid_coco_build import VIEWS

    frames_dir = Path(source_root) / view_key / "frames"
    files = [p for p in frames_dir.glob("*.jpg")]
    files.sort(key=lambda p: (frame_number(p.name), p.name))
    pool = [p for p in files if p.name not in annotated]
    selected = pool[offset::stride]
    return selected


# --------------------------------------------------------------------------- #
# Inference -> candidate COCO json
# --------------------------------------------------------------------------- #


def _to_coco_bbox(x1, y1, x2, y2) -> list[float]:
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]


def pseudo_label_view(
    *,
    build_dir: Path,
    source_root: Path,
    view_key: str,
    weights: str | Path,
    stride: int = 30,
    conf: float = 0.60,
    materialize: bool = False,
) -> tuple[Path, Path]:
    """Label sampled unannotated frames, write candidate + merged jsons.

    Returns (candidate_json, merged_train_json).
    """
    from libreyolo import LibreYOLO
    from .mtid_coco_build import VIEWS

    build_dir = Path(build_dir)
    source_root = Path(source_root)
    _, view_name, _ = VIEWS[view_key]
    images_dir = build_dir / "images"
    ann_dir = build_dir / "annotations"

    # Ground-truth set (train + val) to exclude from the pool.
    annotated = annotated_basenames(build_dir)
    train_data = json.loads((ann_dir / "train.json").read_text())

    pool = list_pool_frames(source_root, view_key, annotated, stride=stride)
    print(f"[{view_name}] unannotated pool sampled: {len(pool)} frames (stride={stride})")
    if not pool:
        raise SystemExit("empty pseudo-label pool - nothing to do")

    model = LibreYOLO(str(weights))

    # Ground-truth already occupies ids; keep pseudo ids strictly after them so a
    # later merge never collides. Start image/annotation ids above any GT id.
    max_img_id = max((img["id"] for img in train_data["images"]), default=0)
    max_ann_id = 0
    for split in ("train", "val"):
        p = ann_dir / f"{split}.json"
        if p.exists():
            d = json.loads(p.read_text())
            max_ann_id = max(max_ann_id, *(a["id"] for a in d["annotations"]), 0)
    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    pseudo_images: list[CocoImage] = []
    pseudo_annotations: list[CocoAnnotation] = []
    n_frames_with_boxes = 0
    n_accepted = 0
    n_rejected_conf = 0

    for i, pool_file in enumerate(pool):
        frame = cv2.imread(str(pool_file), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        H, W = frame.shape[:2]
        result = model(frame)
        boxes = result.boxes
        kept = []
        if boxes is not None:
            xyxy = boxes.xyxy
            bconf = boxes.conf
            bcls = boxes.cls
            for j in range(len(bconf)):
                if float(bconf[j]) < conf:
                    n_rejected_conf += 1
                    continue
                kept.append((xyxy[j], float(bconf[j]), int(bcls[j])))
        if not kept:
            continue  # skip frames with no accepted detection
        n_frames_with_boxes += 1

        # Register the labeled frame under its canonical basename.
        file_name = pool_file.name
        dst = images_dir / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and not dst.is_symlink():
            if materialize:
                shutil.copy2(pool_file, dst)
            else:
                dst.symlink_to(pool_file.resolve())

        pseudo_images.append(
            CocoImage(
                id=next_img_id,
                file_name=file_name,
                width=W,
                height=H,
            )
        )
        for row, c, label in kept:
            x1, y1, x2, y2 = (float(v) for v in (row[0], row[1], row[2], row[3]))
            pseudo_annotations.append(
                CocoAnnotation(
                    id=next_ann_id,
                    image_id=next_img_id,
                    category_id=label + 1,  # label 0..3 -> cat id 1..4
                    bbox=_to_coco_bbox(x1, y1, x2, y2),
                )
            )
            next_ann_id += 1
            n_accepted += 1
        next_img_id += 1

    cats = [CocoCategory(id=i + 1, name=name) for i, name in enumerate(CATEGORY_NAMES)]
    pseudo = {
        "images": [im.model_dump() for im in pseudo_images],
        "annotations": [a.model_dump() for a in pseudo_annotations],
        "categories": [c.model_dump() for c in cats],
    }
    candidate = ann_dir / f"pseudo_{view_name}.json"
    candidate.write_text(json.dumps(pseudo, indent=2))

    # Merge: GT train + pseudo -> train_pseudo json. Image/ann ids already disjoint.
    merged_images = list(train_data["images"])
    merged_annotations = list(train_data["annotations"])
    merged_images.extend(im.model_dump() for im in pseudo_images)
    merged_annotations.extend(a.model_dump() for a in pseudo_annotations)
    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [c.model_dump() for c in cats],
    }
    merged_train = ann_dir / f"train_pseudo_{view_name}.json"
    merged_train.write_text(json.dumps(merged, indent=2))

    print(
        f"[{view_name}] pseudo-labeled frames: {n_frames_with_boxes} "
        f"(accepted boxes {n_accepted}, below-conf dropped {n_rejected_conf})"
    )
    print(f"[{view_name}] wrote {candidate}")
    print(f"[{view_name}] wrote merged training json {merged_train}")
    return candidate, merged_train


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    from .mtid_coco_build import VIEWS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=list(VIEWS), required=True)
    parser.add_argument("--build", type=Path, required=True,
                        help="Stage-0 build dir, e.g. dataset_build/drone")
    parser.add_argument("--source", type=Path, required=True,
                        help="Immutable MTID source root (has Drone/, frames/).")
    parser.add_argument("--weights", type=str, required=True,
                        help="Trained model checkpoint, e.g. .../weights/best.pt")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--materialize", action="store_true",
                        help="Copy labeled frames instead of symlinking")
    args = parser.parse_args()

    candidate, merged = pseudo_label_view(
        build_dir=args.build,
        source_root=args.source,
        view_key=args.view,
        weights=args.weights,
        stride=args.stride,
        conf=args.conf,
        materialize=args.materialize,
    )
    print(f"CANDIDATE_JSON={candidate}")
    print(f"MERGED_TRAIN_JSON={merged}")


if __name__ == "__main__":
    main()
