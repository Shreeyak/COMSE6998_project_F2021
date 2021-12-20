from train_lightning import RotationNetPl, Stage
from dataset_generator.rotation_comparator import RotationComparator

from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_only
from rotation_dataset import RotationDataset
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin

from geodesic_loss import GeodesicDist
import dilated_resnet as drn

class RotationNetPlInference(RotationNetPl):
    def __init__(self):
        super().__init__()
        self.rot_comp = RotationComparator("./test_visuals","./dataset_generator/coordinate.urdf")
        self.total_loss_test = 0.0
        self.total_step_test = 0

    def test_step(self, batch, batch_idx):
        inputs = batch["input"]
        labels = batch["target"]
        bsize = inputs.shape[0]
        labels = labels.reshape(bsize, 3, 3)  # Shape: [B, 3, 3]

        preds = self.model(inputs)  # Shape: [B, 6]

        loss = self.geo_dist(preds, labels)
        loss_degrees = loss*(180.0/np.pi)
        self.total_loss_test += loss_degrees
        self.total_step_test += 1


        for idx,(gt_,pred_) in enumerate(zip(labels.detach().cpu(),preds.detach().cpu())):

            self.rot_comp.compare_rotations(gt_.numpy(),pred_.numpy(),batch_idx*bsize+idx)

        outputs = {
            "loss": loss
        }
        return outputs

    def on_test_epoch_end(self) -> None:

        self.total_loss_test = self.total_loss_test/self.total_step_test

        self.log("inference_test", self.total_loss_test, on_step=False, on_epoch=True)


def main():
    pl.seed_everything(42, workers=True)  # set seed for reproducibility

    dir_root = Path("./logs")
    dir_root.mkdir(exist_ok=True)
    ckp_path = "./checkpoints/epoch=98-step=4553.ckpt"
    model = RotationNetPlInference.load_from_checkpoint(ckp_path)
    wb_logger = pl_loggers.WandbLogger(name=None, id="10z34dh1", entity="rotteam", project="rotenv",
                                       save_dir=str("./logs"))

    data_root_dir ="./dataset/"
    #if not data_root_dir.is_dir():
        #raise ValueError(f"Dir does not exist: {data_root_dir}")

    test_dataset = RotationDataset(data_root_dir+"test" + '/', True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        logger=wb_logger
    )

    # Testing
    _ = trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
