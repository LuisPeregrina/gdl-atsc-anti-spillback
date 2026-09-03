"""Stage 1: train a LibreYOLO9c detection baseline on each MTID view.

Trains one model per view (Drone, Infrastructure) on the native-COCO datasets
produced by tools/mtid_coco_build.py. Each model is initialised from the
LibreYOLO9c COCO-pretrained checkpoint (auto-downloaded on first run) and
fine-tuned to the 4 MTID classes. Training artifacts land under runs/train/
(excluded from git) by LibreYOLO's own trainer.

Usage:
    python tools/train_baseline.py            # train both views
    python tools/train_baseline.py --views Drone
    python tools/train_baseline.py --epochs 20
"""

from __future__ import annotations

import argparse
import pathlib
from pathlib import Path

import torch

torch.serialization.add_safe_globals([pathlib._local.PosixPath])

from libreyolo import LibreYOLO
from libreyolo.training import TrainEpochEvent

from tools.mtid_coco_build import VIEWS

# Per-view data.yaml built by Stage 0.
BUILD_ROOT = Path("dataset_build")
# Pretrained LibreYOLO9c weights (auto-downloaded if absent). "c" = compact size.
MODEL_WEIGHTS = "LibreYOLO9c.pt"


class RunLog:
    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        if event.is_best:
            print(f"[epoch {event.epoch}] new best {event.best_metric}")


def train_view(view_key: str, args: argparse.Namespace) -> dict:
    _, view_name, _ = VIEWS[view_key]
    yaml_path = BUILD_ROOT / view_name / f"{view_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"{yaml_path} missing - run tools/mtid_coco_build.py first"
        )

    print(f"\n=== Training LibreYOLO9c on {view_key} ({view_name}) ===")
    model = LibreYOLO(MODEL_WEIGHTS)  # size "c"; downloads weights on first use

    return model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        lr0=args.lr0,
        pretrained=True,  # transfer from the COCO LibreYOLO9c checkpoint
        seed=args.seed,
        callbacks=RunLog(),
        name=f"mtid_{view_name}",
        project="runs/train",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", nargs="+", choices=list(VIEWS), default=list(VIEWS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for view in args.views:
        results = train_view(view, args)
        print(f"{view} results: {results}")


if __name__ == "__main__":
    main()
