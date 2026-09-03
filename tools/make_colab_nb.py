#!/usr/bin/env python3
"""Emit colab_train_baseline.ipynb (Stage 1 LibreYOLO9c baselines in Colab).

Self-contained: downloads MTID from kagglehub into Colab, materialises the
per-view LibreYOLO-native-COCO datasets with tools/mtid_coco_build.py
(--materialize, i.e. real copies, so it works where symlinks don't), then trains
a LibreYOLO9c baseline per view. Run on a Colab GPU (Runtime -> Change runtime
type -> T4/A100). Long runs: mount Drive at the end to persist weights/run dirs.
"""

import json

CELLS = [
    # ---- install ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%pip install -q kagglehub \"libreyolo[onnx,openvino,fast-eval]\" nncf\n",
            "%pip install -q --upgrade jupyter ipywidgets\n",
        ],
    },
    # ---- clone repo + config ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clone the project (contains tools/mtid_coco_build.py, tools/train_baseline.py)\n",
            "!git clone -q https://github.com/LuisPeregrina/gdl-atsc-anti-spillback.git gdl\n",
            "%cd /content/gdl\n",
        ],
    },
    # ---- imports ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pathlib\n",
            "from pathlib import Path\n",
            "\n",
            "import torch\n",
            "torch.serialization.add_safe_globals([pathlib._local.PosixPath])\n",
            "\n",
            "import kagglehub\n",
            "import yaml\n",
            "from libreyolo import LibreYOLO\n",
            "from libreyolo.training import TrainEpochEvent\n",
            "\n",
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')\n",
        ],
    },
    # ---- download MTID ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "DATASET_NAME = \"andreasmoegelmose/multiview-traffic-intersection-dataset\"\n",
            "dataset_path = kagglehub.dataset_download(DATASET_NAME)\n",
            "print(f\"Dataset downloaded to: {dataset_path}\")\n",
        ],
    },
    # ---- Stage 0: build native-COCO datasets (materialised copies) ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from tools.mtid_coco_build import CocoBuild\n",
            "\n",
            "BUILD_ROOT = Path(\"/content/gdl/dataset_build\")\n",
            "source_root = Path(dataset_path)\n",
            "\n",
            "yaml_paths = {}\n",
            "for view in [\"Drone\", \"Infrastructure\"]:\n",
            "    b = CocoBuild(source_root, BUILD_ROOT, view, seed=42, materialize=True)\n",
            "    yaml_paths[view] = b.build()\n",
            "    print(view, \"->\", yaml_paths[view])\n",
        ],
    },
    # ---- train one view ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Choose which view to train. Run this cell once per view; it persists\n",
            "# checkpoints to runs/train/mtid_<view>/ inside the Colab VM.\n",
            "VIEW = \"Drone\"  # or \"Infrastructure\"\n",
            "EPOCHS = 30\n",
            "BATCH = 32\n",
            "IMGSZ = 640\n",
            "LEARN_RATE = 3e-4\n",
            "\n",
            "model = LibreYOLO(\"LibreYOLO9c.pt\")\n",
            "\n",
            "class RunLog:\n",
            "    def on_train_epoch_end(self, event: TrainEpochEvent):\n",
            "        if event.is_best:\n",
            "            print(f\"[epoch {event.epoch}] new best {event.best_metric}\")\n",
            "\n",
            "results = model.train(\n",
            "    data=str(yaml_paths[VIEW]),\n",
            "    epochs=EPOCHS,\n",
            "    imgsz=IMGSZ,\n",
            "    batch=BATCH,\n",
            "    workers=2,\n",
            "    lr0=LEARN_RATE,\n",
            "    pretrained=True,\n",
            "    seed=42,\n",
            "    callbacks=RunLog(),\n",
            "    name=f\"mtid_{VIEW.lower()}\",\n",
            "    project=\"runs/train\",\n",
            ")\n",
            "print(results)\n",
        ],
    },
    # ---- persist to Drive ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "from google.colab import drive\n",
            "\n",
            "try:\n",
            "    drive.mount('/content/drive')\n",
            "    drive_root = Path('/content/drive/MyDrive')\n",
            "except ModuleNotFoundError:\n",
            "    drive_root = Path.home()  # local fallback\n",
            "\n",
            "target = drive_root / 'gdl-atsc-anti-spillback' / 'baseline_runs'\n",
            "target.mkdir(parents=True, exist_ok=True)\n",
            "src = Path('runs/train')\n",
            "import shutil\n",
            "for d in src.glob('mtid_*'):\n",
            "    dst = target / d.name\n",
            "    if dst.exists():\n",
            "        shutil.rmtree(dst)\n",
            "    shutil.copytree(d, dst)\n",
            "    print('saved', d, '->', dst)\n",
        ],
    },
]

nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("colab_train_baseline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote colab_train_baseline.ipynb")
