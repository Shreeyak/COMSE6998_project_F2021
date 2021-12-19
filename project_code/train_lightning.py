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
from geodesic_loss import geodesic_dist


class Stage(Enum):
    """Which stage of training we're in. Used primarily to set the string for logging"""

    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


class RotationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8*64*64, 4*32*32)
        self.layer2 = nn.Linear(4*32*32, 2*16*16)
        self.layer3 = nn.Linear(2 * 16 * 16, 1*8*8)
        self.layer4 = nn.Linear(1 * 8 * 8, 4*4)
        self.layer5 = nn.Linear(4*4, 9)
        self.model = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5)

    def forward(self, x):
        x = self.model(x)
        bsize = x.shape[0]
        x = x.reshape((bsize, 3, 3))
        q, _ = torch.linalg.qr(x)
        return q


class RotationNetPl(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()  # Will save args to hparams attr. Also allows upload of config to wandb.

        self.model = RotationNet()

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

        loss = geodesic_dist(preds, labels)
        self.log(f"{stage.value}/loss", loss)

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
                                     weight_decay=0)
        ret_opt = {"optimizer": optimizer}
        return ret_opt


def main():
    pl.seed_everything(42, workers=True)  # set seed for reproducibility

    dir_root = Path("./logs")
    dir_root.mkdir(exist_ok=True)
    wb_logger = pl_loggers.WandbLogger(name=None, id=None, entity="cleargrasp2", project="rotations",
                                       save_dir=str("./logs"))

    callbacks = [
        ModelCheckpoint(save_top_k=2, monitor="Train/loss", mode="min"),  #
    ]
    model = RotationNetPl()

    train_dir = Path("./dataset/train1")
    if not train_dir.is_dir():
        raise ValueError(f"Dir does not exist: {train_dir}")

    train_dataset = RotationDataset(str(train_dir) + '/', True)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0, drop_last=True)

    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=callbacks,
        # default_root_dir=str(default_root_dir),
        strategy=DDPPlugin(find_unused_parameters=False),
        gpus=0,
        precision=32,
        max_epochs=10,
        log_every_n_steps=50,
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
