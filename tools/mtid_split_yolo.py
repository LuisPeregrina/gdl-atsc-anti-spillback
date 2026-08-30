"""Split the MTID COCO jsons into train/val/test YOLO-format subsets.

The MTID dataset ships two monolithic COCO annotation files
(drone-mscoco.json, infrastructure-mscoco.json) with no train/val/test
split. This generator merges both views into a single dataset and writes
standard YOLO-format artifacts, which LibreYOLO consumes via its YOLODataset
path:

  * A YOLO label .txt file beside every image (x.jpg -> x.txt), converted
    from the COCO bounding-box annotations (boxes clipped to the image).
  * train/val/test.list.txt image-list files in the dataset root, one line
    per image path (relative to the root).

Every image lives in one shared source folder (Drone/, Infrastructure/), so
the split is expressed with the image-list text files instead of separate
image directories. The data.yaml sets ``path`` to the dataset root and points
train/val/test at the .list.txt files by bare name; LibreYOLO resolves each
image (relative to the list file's directory) and derives its label path by
swapping the extension (x.jpg -> x.txt, same directory).

Design decisions (matching the project's intent):
  * One combined dataset (drone + infrastructure merged).
  * Zero-annotation images are KEPT (empty label files).
  * Content-identical duplicate frames are DEDUPLICATED (kept once) so the
    same pixels never span two splits.
  * Categories are filtered to the four that actually carry annotations:
    car, lorry, bus, bicycle.
  * Deterministic, seeded split (default 80/10/10; ratios configurable).
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class CocoImage(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    file_name: str
    width: int
    height: int
    license: int | None = None
    flickr_url: str | None = None
    coco_url: str | None = None
    date_captured: str | None = None


class CocoAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float | None = None
    segmentation: list[list[float]] | list[float] | None = None
    iscrowd: int | None = None
    ignore: int | None = None


class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str | None = None


class CocoDataset(BaseModel):
    model_config = ConfigDict(extra="allow")

    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory] | None = None


SplitPaths = dict[str, Path]
SplitImageLists = dict[str, list[str]]


KEEP_CATEGORIES = {
    2: "bicycle",
    3: "car",
    # 4: "motorcycle",  # Not present in the dataset
    6: "bus",
    8: "lorry",  # this dataset annotates as lorry instead of truck
    # TODO: Emergency vehicles, need different / complementary dataset
}

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# COCO category id -> YOLO label index (insertion order of KEEP_CATEGORIES).
CATEGORY_INDEX = {cid: i for i, cid in enumerate(KEEP_CATEGORIES)}


def load(dataset_root: Path) -> tuple[list[CocoImage], list[CocoAnnotation]]:
    """Load and merge both COCO jsons into (images, annotations).

    Images are re-keyed to a single non-overlapping id range across the two
    files (they use overlapping id ranges), and every annotation's image_id is
    remapped to the new value. Only kept categories are returned.
    """
    images: list[CocoImage] = []
    annotations: list[CocoAnnotation] = []
    cat_ids = set(KEEP_CATEGORIES)

    ann_id_offset = 0
    img_id_offset = 0
    for json_name in ("drone-mscoco.json", "infrastructure-mscoco.json"):
        path = dataset_root / json_name
        if not path.exists():
            raise FileNotFoundError(path)
        data = CocoDataset.model_validate_json(path.read_text())

        # Re-key image ids so they stay unique across both files (the two
        # COCO files use overlapping id ranges). Must also remap every
        # annotation's image_id to the new value.
        old_to_new: dict[int, int] = {}
        for img in data.images:
            new_img = CocoImage(**img.model_dump())
            new_id = img_id_offset
            img_id_offset += 1
            old_to_new[img.id] = new_id
            new_img.id = new_id
            images.append(new_img)

        for ann in data.annotations:
            if ann.category_id not in cat_ids:
                continue
            new_ann = CocoAnnotation(**ann.model_dump())
            new_ann.id = ann_id_offset
            ann_id_offset += 1
            new_ann.image_id = old_to_new[ann.image_id]
            annotations.append(new_ann)

    return images, annotations


def split(
    images: list[CocoImage],
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[int, str]:
    """Assign every image id to a split, deterministically.

    Shuffles with a seeded RNG, then walks ``SPLITS`` in order filling each by
    ``ratio * n`` images; the rounding remainder is swept into ``test``.
    Returns a mapping of image id -> split name.
    """
    rng = random.Random(seed)
    images = list(images)
    rng.shuffle(images)

    ratios = ratios or RATIOS
    assignments: dict[int, str] = {}
    cursor = 0
    for split_name in SPLITS:
        n = round(len(images) * ratios[split_name])
        chunk = images[cursor : cursor + n]
        cursor += n
        for img in chunk:
            assignments[img.id] = split_name
    # Sweep the remainder (rounding) into test.
    for img in images[cursor:]:
        assignments[img.id] = "test"
    return assignments


def yolo_line(ann: CocoAnnotation, img: CocoImage) -> str | None:
    """Convert one COCO annotation to a YOLO label line.

    YOLO format: ``class cx cy w h``, all normalized to [0, 1]. COCO bbox is
    ``[x, y, width, height]`` in absolute pixels. The class index comes from
    ``CATEGORY_INDEX`` (KEEP_CATEGORIES insertion order).

    The box is first clipped to the image bounds so the normalized values stay
    within [0, 1] (the source COCO has boxes extending past image edges).
    Returns ``None`` if clipping leaves a degenerate (zero-area) box.
    """
    x, y, w, h = ann.bbox
    W, H = img.width, img.height
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    cw, ch = x1 - x0, y1 - y0
    cls = CATEGORY_INDEX[ann.category_id]
    return (
        f"{cls} "
        f"{(x0 + cw / 2) / W:.6f} "
        f"{(y0 + ch / 2) / H:.6f} "
        f"{cw / W:.6f} "
        f"{ch / H:.6f}"
    )


def dedupe_by_content(
    dataset_root: Path,
    images: list[CocoImage],
    annotations: list[CocoAnnotation],
) -> tuple[list[CocoImage], list[CocoAnnotation], list[CocoImage]]:
    """Drop content-identical duplicate images, keeping the first occurrence.

    The MTID download duplicates some frames byte-for-byte (e.g.
    ``Infrastructure/0/seq3-infra_0001000.jpg`` ==
    ``Infrastructure/1000/seq3-infra_0001000.jpg``). If such twins land in
    different splits, LibreYOLO flags ``splits.leakage_exact``. Dedupe by file
    content hash (SHA-1, matching the doctor's exact-leakage key) so the same
    pixels exist in exactly one split. The duplicate's annotations are dropped
    with it. Returns (kept images, kept annotations, dropped images).
    """
    seen: set[str] = set()
    kept_ids: set[int] = set()
    kept_images: list[CocoImage] = []
    dropped_images: list[CocoImage] = []
    for img in images:
        digest = hashlib.sha1((dataset_root / img.file_name).read_bytes()).hexdigest()
        if digest in seen:
            dropped_images.append(img)
            continue
        seen.add(digest)
        kept_ids.add(img.id)
        kept_images.append(img)

    kept_annotations = [ann for ann in annotations if ann.image_id in kept_ids]
    dropped = len(images) - len(kept_images)
    if dropped:
        print(f"deduplicated {dropped} content-identical image(s)")
    return kept_images, kept_annotations, dropped_images


def write_yolo_labels(
    dataset_root: Path,
    images: list[CocoImage],
    annotations: list[CocoAnnotation],
    dropped_images: list[CocoImage] | None = None,
) -> None:
    """Write a YOLO label .txt next to each image (x.jpg -> x.txt).

    The source image dirs are the label home in YOLO format; LibreYOLO finds
    a label by swapping the image extension, so the label must share the
    image's directory and basename. Annotation-free images get an empty file.
    Any stale label file for a dropped (deduplicated) image is removed so no
    orphan label (``files.orphan_label``) lingers from a previous run.
    """
    ann_by_img: dict[int, list[CocoAnnotation]] = {}
    for ann in annotations:
        ann_by_img.setdefault(ann.image_id, []).append(ann)

    written = 0
    for img in images:
        img_path = dataset_root / img.file_name
        label_path = img_path.with_suffix(".txt")
        lines: list[str] = []
        for ann in ann_by_img.get(img.id, []):
            line = yolo_line(ann, img)
            if line is not None:
                lines.append(line)
        label_path.write_text("".join(f"{line}\n" for line in lines))
        written += 1

    removed = 0
    for img in dropped_images or []:
        label_path = (dataset_root / img.file_name).with_suffix(".txt")
        if label_path.exists():
            label_path.unlink()
            removed += 1
    print(
        f"wrote {written} YOLO label files beside images"
        + (f"; removed {removed} stale label(s)" if removed else "")
    )


def write_split_lists(
    dataset_root: Path,
    images: list[CocoImage],
    assignments: dict[int, str],
) -> SplitPaths:
    """Write one image-list text file per split, into the dataset root.

    Each ``{split}.list.txt`` lists the image paths (relative to the dataset
    root) for that split, one per line. Paths stay relative so the yaml can
    reference the lists by bare name under ``path:``; LibreYOLO resolves each
    entry relative to the list file's directory (= the dataset root).
    """
    by_split: SplitImageLists = {"train": [], "val": [], "test": []}
    for img in images:
        by_split[assignments[img.id]].append(img.file_name)

    paths_dict: dict[str, Path] = {}
    for s in SPLITS:
        path = dataset_root / f"{s}.list.txt"
        path.write_text("".join(f"{p}\n" for p in sorted(by_split[s])))
        print(f"wrote {path} ({len(by_split[s])} images)")
        paths_dict[s] = path
    return paths_dict  # type: ignore[return-value]


def write_data_yaml(
    dataset_root: Path,
    out_path: Path,
    list_paths: SplitPaths,
) -> Path:
    """Write the LibreYOLO data.yaml pointing at the image-list files.

    ``path`` is the dataset root; ``train/val/test`` point at the .list.txt
    files by bare relative name. Names are keyed by label index (0..3),
    matching KEEP_CATEGORIES insertion order. There is no ``annotations:``
    block: LibreYOLO loads labels via YOLODataset (image list + derived label
    paths).
    """
    names = {label: name for label, (_cid, name) in enumerate(KEEP_CATEGORIES.items())}
    content = (
        "# Autogenerated by tools/split_yolo.py. Do not edit.\n"
        f"# MTID (Multiview Traffic Intersection Dataset) - combined drone + infra.\n"
        f"# YOLO-format image-list mode: train/val/test point at {{split}}.list.txt\n"
        f"# files enumerating the exact images per split (all images share one folder).\n"
        f"# Each image's label is x.txt next to x.jpg.\n"
        f"path: {dataset_root}\n"
        f"\n"
        f"train: {list_paths['train'].name}\n"
        f"val: {list_paths['val'].name}\n"
        f"test: {list_paths['test'].name}\n"
        f"\n"
        f"nc: {len(names)}\n"
        f"names:\n"
    )
    for label, name in names.items():
        content += f"  {label}: {name}\n"
    out_path.write_text(content)
    print(f"wrote {out_path}")
    return out_path


def split_dataset(
    dataset_root: str | Path,
    out_dir: str | Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Path:
    """Build the YOLO-format split dataset and return the data.yaml path.

    Writes YOLO label files beside the source images and the split list files
    (train/val/test.list.txt) into the dataset root. ``mtid.yaml`` lives next to
    those artifacts by default, in the dataset directory itself.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)

    ratios = RATIOS | {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    if ratios["test"] < 0:
        raise ValueError(
            f"train_ratio + val_ratio ({train_ratio} + {val_ratio}) must be <= 1.0"
        )

    out_path = Path(out_dir) if out_dir is not None else root
    out_path.mkdir(parents=True, exist_ok=True)

    images, annotations = load(root)
    # Drop content-identical duplicates before splitting so the same pixels
    # never land in two different splits (LibreYOLO would flag leakage_exact).
    images, annotations, dropped = dedupe_by_content(root, images, annotations)
    assignments = split(images, seed, ratios)

    counts = {s: 0 for s in SPLITS}
    for img in images:
        counts[assignments[img.id]] += 1
    total = sum(counts.values())
    print(
        f"total: {total} images across {', '.join(SPLITS)} "
        f"({', '.join(f'{s}: {counts[s]}' for s in SPLITS)})"
    )

    # Convert COCO annotations -> YOLO label files, then write the image lists
    # and the data.yaml that points LibreYOLO at those lists.
    write_yolo_labels(root, images, annotations, dropped)
    list_paths = write_split_lists(root, images, assignments)
    yaml_path = write_data_yaml(root, out_path / "mtid.yaml", list_paths)

    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root with drone-mscoco.json and infrastructure-mscoco.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for mtid.yaml (default: dataset root)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic split"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images for train (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of images for val (default: 0.1); test = 1 - train - val",
    )
    args = parser.parse_args()

    yaml_path = split_dataset(
        args.root,
        args.out,
        args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Machine-friendly line so callers can capture the yaml path.
    print(f"DATA_YAML={yaml_path}")


if __name__ == "__main__":
    main()
