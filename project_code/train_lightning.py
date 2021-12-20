import datetime
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_only
from rotation_dataset import RotationDataset
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from geodesic_loss import GeodesicDist


class Stage(Enum):
    """Which stage of training we're in. Used primarily to set the string for logging"""

    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


class OrthoMatrix:
    """Functions to create an Orthonormal Matrix from the prediction of the model.
    Borrowed from code of paper "On The Continuity of Rotation Representations in Neural Networks"

    Ref:
        https://github.com/papagina/RotationContinuity

        Zhou, Yi, et al. "On the continuity of rotation representations in neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    """

    # batch*n
    @staticmethod
    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        eps = torch.tensor([1e-8], dtype=torch.float, device=v.device, requires_grad=True)
        v_mag = torch.max(v_mag, eps)
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if (return_mag == True):
            return v, v_mag[:, 0]
        else:
            return v

    # u, v batch*n
    @staticmethod
    def cross_product(u, v):
        batch = u.shape[0]
        # print (u.shape)
        # print (v.shape)
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

        return out

    def compute_rotation_matrix_from_ortho6d(self, ortho6d):
        x_raw = ortho6d[:, 0:3]  # batch*3
        y_raw = ortho6d[:, 3:6]  # batch*3

        x = self.normalize_vector(x_raw)  # batch*3
        z = self.cross_product(x, y_raw)  # batch*3
        z = self.normalize_vector(z)  # batch*3
        y = self.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2)  # batch*3*3
        return matrix


class RotationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8*64*64, 4*32*32)
        self.layer2 = nn.Linear(4*32*32, 2*16*16)
        self.layer3 = nn.Linear(2 * 16 * 16, 1*8*8)
        self.layer4 = nn.Linear(1 * 8 * 8, 4*4)
        self.layer5 = nn.Linear(4*4, 6)
        self.model = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5)

        self.ortho_mat = OrthoMatrix()

    def forward(self, x):
        x = self.model(x)  # Shape: [B, 6]
        rot_mat = self.ortho_mat.compute_rotation_matrix_from_ortho6d(x)  # Shape: [B, 3, 3]
        return rot_mat


class RotationNetPl(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()  # Will save args to hparams attr. Also allows upload of config to wandb.

        self.model = RotationNet()

        self.geo_dist = GeodesicDist(reduction="mean")

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Shape: [B, N]
        """
        out = self.model(inputs)
        return out

    def _step(self, batch, stage: Stage):
        inputs = batch["input"]
        labels = batch["target"]
        bsize = inputs.shape[0]
        inputs = inputs.reshape(bsize, -1)  # Shape: [B, N]
        labels = labels.reshape(bsize, 3, 3)  # Shape: [B, 3, 3]

        # preds = self(inputs)
        preds = self.model(inputs)

        loss = self.geo_dist(preds, labels)
        self.log(f"{stage.value}/loss", loss, on_step=False, on_epoch=True)

        outputs = {
            "loss": loss
        }
        return outputs

    def training_step(self, batch, batch_idx):
        """Defines the train loop. It is independent of forward().
        Don’t use any cuda or .to(device) calls in the code. PL will move the tensors to the correct device.
        """
        return self._step(batch, Stage.TRAIN)

    # def validation_step(self, batch, batch_idx):
    #     return self._step(batch, Stage.VAL)
    #
    # def test_step(self, batch, batch_idx):
    #     return self._step(batch, Stage.TEST)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=1e-3,
                                     weight_decay=0.01)
        ret_opt = {"optimizer": optimizer}
        return ret_opt


def main():
    pl.seed_everything(42, workers=True)  # set seed for reproducibility

    dir_root = Path("./logs")
    dir_root.mkdir(exist_ok=True)
    wb_logger = pl_loggers.WandbLogger(name=None, id=None, entity="rotteam", project="rotenv",
                                       save_dir=str("./logs"))

    callbacks = [
        ModelCheckpoint(save_top_k=2, monitor="Train/loss", mode="min"),  #
    ]
    model = RotationNetPl()

    train_dir = Path("./dataset/train1")
    if not train_dir.is_dir():
        raise ValueError(f"Dir does not exist: {train_dir}")

    train_dataset = RotationDataset(str(train_dir) + '/', True)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=8, drop_last=True)

    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=callbacks,
        # default_root_dir=str(default_root_dir),
        #strategy=DDPPlugin(find_unused_parameters=False),
        gpus=1,
        precision=32,
        max_epochs=10,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        overfit_batches=0.0,
    )

    # Run Training
    trainer.fit(model, train_dataloaders=train_loader)

    # _ = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
