import shutil
from pathlib import Path
from time import sleep

import kagglehub
import yaml
from libreyolo import LibreYOLO
from libreyolo.training import TrainEndEvent, TrainEpochEvent, TrainStartEvent, TrainExceptionEvent

from tools.mtid_split_yolo import split_dataset

MODEL_NAME = "LibreFOMOs-point"
#!curl -L https://huggingface.co/LibreYOLO/{MODEL_NAME}/resolve/main/{MODEL_NAME}.pt -o weights/{MODEL_NAME}.pt

DATASET_PATH = Path.cwd() / "dataset"
DATASET_NAME = "andreasmoegelmose/multiview-traffic-intersection-dataset"
EPOCHS = 100

# Configs per model
models = yaml.safe_load(Path("models.yaml").read_text())["models"]
model = next((m for m in models if m["name"] == MODEL_NAME), None)
IMAGE_SIZE = int(model["image_size"])
BATCH_SIZE = int(model["batch_size"])
dataset_path = kagglehub.dataset_download(DATASET_NAME, output_dir=str(DATASET_PATH))
MINUTES = 1


yaml_path = split_dataset(dataset_path)

# Logger
class RunLog:
    def copy_last(self, event: TrainEpochEvent) -> None:
        fname = "last.pt"
        event_last_pt = Path(event.save_dir) / "weights" / fname
        if not event_last_pt.exists():
            print(f"Warning: {event_last_pt} does not exist, skipping copy.")
            return
        print(f"Copying {event_last_pt} to {fname}")
        shutil.copy(event_last_pt, fname)

 
    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        if event.is_best:
            print(f"new best at epoch {event.epoch}: {event.best_metric}")

        self.copy_last(event)
        # print(f"Sleeping {MINUTES} minutes to cool down...")
        # sleep(60*MINUTES)
 


resuming = False

model = LibreYOLO(f"{MODEL_NAME}.pt" if not resuming else "last.pt")
last_weights_path = None
model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=0,
        callbacks=RunLog(),
        resume=resuming
    )
