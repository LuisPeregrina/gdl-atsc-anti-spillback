"""Stage 0: build per-view LibreYOLO-native COCO datasets from the immutable MTID source.

Goal
----
MTID ships two monolithic, un-split COCO jsons (``drone-mscoco.json``,
``infrastructure-mscoco.json``) covering only the ~3k human-annotated frames per
view. The raw full-sequence frames live in the immutable ``frames/`` folder per
view. We never edit the source; this generator materialises a *separate* build
tree that LibreYOLO can train on in native-COCO-JSON mode (the format LibreYOLO
consumes directly, see its data-config loader ``load_data_config``/``COCODataset``).

For each view it writes::

    dataset_build/<view>/
      images/                    # symlinks -> source frames/ (immutable source untouched)
          seq3-<view>_NNNNNNN.jpg
      annotations/
          train.json             # COCO, 4 classes
          val.json
      <view>.yaml                # LibreYOLO native-COCO data.yaml:
                                 #   path, train/val image dir, annotations:{train,val}, names

Why symlinks instead of copies
-----------------------------
Native-COCO mode pairs one image directory with each COCO json and resolves every
image as ``path/images/<file_name>``. The source annotated frames live scattered
across numbered folders and the raw pool in ``frames/``; rather than copy 5.7k+
images (duplicating ~GBs) we symlink only the frames each json needs. Symlinking
keeps the source strictly read-only while presenting LibreYOLO a clean
single-folder-per-view image area. Use ``--materialize`` to copy instead of
symlink (e.g. for a Docker build context or a filesystem without symlink support).

Frame keying / the pseudo-label hook
------------------------------------
Every frame in ``frames/`` has a unique basename ``seq3-<view>_NNNNNNN.jpg``
(frame number is the identity). The annotated jsons reference exactly these
filenames. This makes the annotated-set minus the raw pool trivially computable
and stable across Stage 2 (pseudo-labelling): Stage 2 will add more symlinks to
the same ``images/`` area and emit a second, merged COCO json, reusing the exact
same frame-keying so ground truth and pseudo-labels never collide.

Only the four MTID classes that carry annotations are kept: bicycle, car, bus,
lorry (MTID calls trucks "lorry"). Class order [bicycle, car, bus, lorry] matches
the repo's existing YOLO mapping. Category ids are re-keyed to 1..4 and image /
annotation ids re-keyed to 1..n (pycocotools-friendly, no id 0).

Split policy
------------
Deterministic, seeded shuffle of the annotated frames into train/val
(default 90/10). Annotation-free frames are retained (empty objects list) so the
json faithfully represents the annotated set. Temporally adjacent frames may
land in different splits (they are distinct content, so no byte-exact leakage;
LibreYOLO's doctor flags only exact-duplicate leakage).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# --------------------------------------------------------------------------- #
# COCO data models (lenient: ignore unknown source fields we do not emit)
# --------------------------------------------------------------------------- #


class CocoImage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    file_name: str
    width: int
    height: int


class CocoAnnotation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    iscrowd: int = 0


class CocoCategory(BaseModel):
    id: int
    name: str


class CocoDataset(BaseModel):
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# view-key (source dir) -> (json filename, build view name, sequence basename)
VIEWS = {
    "Drone": ("drone-mscoco.json", "drone", "seq3-drone"),
    "Infrastructure": ("infrastructure-mscoco.json", "infra", "seq3-infra"),
}

# Source MTID category id -> (new label index 0..3). Dropped categories (the
# 80-class COCO table has only these four populated) are simply filtered out.
# Order defines both the json category ids (1..4) and the yaml names index.
KEEP_CATEGORY_IDS = [2, 3, 6, 8]  # bicycle, car, bus, lorry
CATEGORY_NAMES = ["bicycle", "car", "bus", "lorry"]

DEFAULT_RATIOS = {"train": 0.9, "val": 0.1}


class CocoBuild:
    """Materialise one view's LibreYOLO-native COCO dataset into a build dir."""

    def __init__(
        self,
        source_root: Path,
        build_root: Path,
        view_key: str,
        *,
        seed: int = 42,
        materialize: bool = False,
    ):
        if view_key not in VIEWS:
            raise ValueError(f"Unknown view {view_key!r}; choose from {list(VIEWS)}")
        self.source_root = Path(source_root)
        self.build_root = Path(build_root)
        self.view_key = view_key
        json_name, self.view_name, self.seq = VIEWS[view_key]
        self.seed = seed
        self.materialize = materialize

        self.source_json = self.source_root / json_name
        if not self.source_json.exists():
            raise FileNotFoundError(self.source_json)
        self.frames_dir = self.source_root / view_key / "frames"
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(self.frames_dir)

        # Keep only the four populated MTID categories, remapped to 1..4.
        self.keep_ids: set[int] = set(KEEP_CATEGORY_IDS)
        self.label_of: dict[int, int] = {
            cat_id: i for i, cat_id in enumerate(KEEP_CATEGORY_IDS)
        }

        # Build-tree paths.
        self.view_dir = self.build_root / self.view_name
        self.images_dir = self.view_dir / "images"
        self.ann_dir = self.view_dir / "annotations"
        self.train_json = self.ann_dir / "train.json"
        self.val_json = self.ann_dir / "val.json"
        self.yaml_path = self.view_dir / f"{self.view_name}.yaml"

    # -- loading ----------------------------------------------------------- #

    def _load(self) -> tuple[list[CocoImage], list[CocoAnnotation]]:
        data = json.loads(self.source_json.read_text())

        # Re-key images to 1..n (pycocotools-friendly, no id 0) and record the
        # old->new mapping so annotations can be remapped to their image.
        images: list[CocoImage] = []
        old_to_new_img: dict[int, int] = {}
        for new_id, img in enumerate(data["images"], start=1):
            old_to_new_img[img["id"]] = new_id
            images.append(
                CocoImage(
                    id=new_id,
                    file_name=img["file_name"],
                    width=int(img["width"]),
                    height=int(img["height"]),
                )
            )

        annotations: list[CocoAnnotation] = []
        for ann_id, ann in enumerate(data["annotations"], start=1):
            if ann["category_id"] not in self.keep_ids:
                continue
            new_img_id = old_to_new_img.get(ann["image_id"])
            if new_img_id is None:
                continue  # orphan annotation (image not in this json)
            annotations.append(
                CocoAnnotation(
                    id=ann_id,
                    image_id=new_img_id,
                    category_id=int(ann["category_id"]),
                    bbox=[float(v) for v in ann["bbox"]],
                )
            )
        return images, annotations

    # -- split ------------------------------------------------------------- #

    def _split(
        self, images: list[CocoImage], ratios: dict[str, float]
    ) -> dict[str, set[int]]:
        """Deterministically assign each image id to train/val (seeded shuffle).

        Returns {split: {image_id, ...}}.
        """
        rng = random.Random(self.seed)
        ordered = sorted(images, key=lambda im: im.file_name)
        rng.shuffle(ordered)
        n_train = round(len(ordered) * ratios["train"])
        train_ids = {im.id for im in ordered[:n_train]}
        val_ids = {im.id for im in ordered[n_train:]}
        return {"train": train_ids, "val": val_ids}

    # -- image materialisation -------------------------------------------- #

    def _link_image(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if self.materialize:
            import shutil

            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

    def _canonical_src(self, basename: str) -> Path:
        """Return the source path of a frame by its basename.

        ``frames/`` holds exactly one file per frame number, which is the
        canonical source for both the annotated jsons and the (later) pseudo-label
        pool. This keeps a single source of truth per frame.
        """
        src = self.frames_dir / basename
        if not src.is_file():
            raise FileNotFoundError(
                f"{self.view_key}: source frame {src} missing; "
                f"cannot reference {basename!r}"
            )
        return src

    def _resolve_annotated(
        self,
        images: list[CocoImage],
        annotations: list[CocoAnnotation],
    ) -> tuple[list[CocoImage], list[CocoAnnotation]]:
        """Resolve each annotated image to its canonical ``frames/`` file.

        The source numbered folders (``0/``, ``1000/``, ...) hold byte-for-byte
        copies of frames that also exist in ``frames/``, and occasionally a stray
        duplicate whose basename collides with another entry. We keep only the
        image whose bytes match the canonical ``frames/<basename>`` file; any
        stray copy (same basename, different content) is dropped so each retained
        image maps to exactly one frame number and never to a shared symlink.
        """
        import hashlib

        def sha1(path: Path) -> str:
            h = hashlib.sha1()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        # Pre-hash the canonical frames/ files we will compare against.
        frame_hash: dict[str, str] = {}
        kept_images: list[CocoImage] = []
        kept_ids: set[int] = set()
        seen_source: set[str] = set()

        for im in images:
            basename = Path(im.file_name).name
            if basename not in frame_hash:
                canon = self._canonical_src(basename)
                frame_hash[basename] = sha1(canon)

            src = self.source_root / im.file_name
            if not src.is_file():
                continue
            if sha1(src) != frame_hash[basename]:
                # Stray duplicate: content differs from the canonical frames/ frame.
                print(f"  drop {im.file_name} (content differs from canonical frames/ copy)")
                continue
            if basename in seen_source:
                # Byte-identical duplicate already kept once under this basename.
                print(f"  drop {im.file_name} (duplicate content)")
                continue
            seen_source.add(basename)
            kept_ids.add(im.id)
            kept_images.append(im)

        kept_annotations = [a for a in annotations if a.image_id in kept_ids]
        return kept_images, kept_annotations

    # -- serialisation ----------------------------------------------------- #

    def _write_coco(
        self,
        images: list[CocoImage],
        annotations: list[CocoAnnotation],
        split: str,
        split_ids: set[int],
    ) -> None:
        """Write one COCO json for a subset of images.

        file_name is the bare basename; LibreYOLO's COCODataset resolves it under
        ``path/<image-dir>/<file_name>`` so every image is also symlinked into
        the shared ``images/`` folder.
        """
        split_images = [im for im in images if im.id in split_ids]
        anns = [a for a in annotations if a.image_id in split_ids]
        ann_img_ids = {a.image_id for a in anns}

        # Remap json category ids to 1..4 in the [bicycle,car,bus,lorry] order.
        cats = [
            CocoCategory(id=i + 1, name=name)
            for i, name in enumerate(CATEGORY_NAMES)
        ]
        # The source kept category ids (2,3,6,8) map to labels 0..3; store label+1
        # as the COCO category id so category 1==label 0 (matches names index).
        serialised = [
            a.model_copy(update={"category_id": self.label_of[a.category_id] + 1})
            for a in anns
        ]

        payload = {
            "images": [
                im.model_copy(update={"file_name": Path(im.file_name).name}).model_dump()
                for im in split_images
            ],
            "annotations": [a.model_dump() for a in serialised],
            "categories": [c.model_dump() for c in cats],
        }
        out = self.ann_dir / f"{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        n_empty = sum(1 for im in split_images if im.id not in ann_img_ids)
        print(
            f"{self.view_name}/{split}: {len(split_images)} images, "
            f"{len(serialised)} annotations, {n_empty} annotation-free"
        )

    def _write_yaml(self) -> None:
        names = {i: name for i, name in enumerate(CATEGORY_NAMES)}
        root = self.view_dir.resolve()
        content = (
            f"# Autogenerated by tools/mtid_coco_build.py. Do not edit.\n"
            f"# {self.view_key} view, MTID, LibreYOLO native-COCO-JSON mode.\n"
            f"# Images live in images/ (symlinks into the immutable kagglehub cache).\n"
            f"path: {root}\n"
            f"train: images\n"
            f"val: images\n"
            f"annotations:\n"
            f"  train: annotations/train.json\n"
            f"  val: annotations/val.json\n"
            f"nc: {len(names)}\n"
            f"names:\n"
        )
        for label, name in names.items():
            content += f"  {label}: {name}\n"
        self.yaml_path.write_text(content)
        print(f"wrote {self.yaml_path}")

    # -- orchestration ----------------------------------------------------- #

    def build(self, ratios: dict[str, float] | None = None) -> Path:
        ratios = ratios or DEFAULT_RATIOS
        images, annotations = self._load()
        images, annotations = self._resolve_annotated(images, annotations)

        split_ids = self._split(images, ratios)

        # Materialise images referenced by the jsons into the shared images/ dir.
        for im in images:
            self._link_image(
                self._canonical_src(Path(im.file_name).name),
                self.images_dir / Path(im.file_name).name,
            )

        self._write_coco(images, annotations, "train", split_ids["train"])
        self._write_coco(images, annotations, "val", split_ids["val"])
        self._write_yaml()
        return self.yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="MTID kagglehub source root (dir containing drone-mscoco.json, "
        "Drone/, Infrastructure/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Build root; per-view trees are written under it.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=list(VIEWS),
        default=list(VIEWS),
        help="Which views to build (default: both).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the train/val split."
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Copy images instead of symlinking them (for Docker contexts etc.).",
    )
    args = parser.parse_args()

    for view in args.views:
        builder = CocoBuild(
            args.source,
            args.out,
            view,
            seed=args.seed,
            materialize=args.materialize,
        )
        yaml_path = builder.build()
        print(f"DATA_YAML={yaml_path}")


if __name__ == "__main__":
    main()
