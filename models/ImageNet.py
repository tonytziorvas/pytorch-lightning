from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10

from models.ConvBlock import ConvBlock
from models.FcBlock import FcBlock


class ImageNet(pl.LightningModule):
    def __init__(self, num_classes, batch_size, transforms, target_transforms):
        super().__init__()

        self.batch_size = batch_size
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # self.save_hyperparameters(logger=False, ignore=["transforms"])

        layers = [
            ConvBlock(
                in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False
            ),
            ConvBlock(
                in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False
            ),
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False
            ),
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            ),
            nn.Flatten(start_dim=1),
            FcBlock(in_channels=128 * 4 * 4, out_channels=128),
            FcBlock(in_channels=128, out_channels=64, use_norm=True),
            FcBlock(in_channels=64, out_channels=num_classes),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)

        self.log_dict(
            {"train_loss": loss, "train_acc": acc},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)

        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=0.005)

    def train_dataloader(self) -> Any:
        dataset = CIFAR10(
            root="data/raw",
            download=True,
            train=True,
            transform=self.transforms,
        )

        return DataLoader(
            dataset, batch_size=64, num_workers=4, shuffle=True, persistent_workers=True
        )

    def test_dataloader(self) -> Any:
        dataset = CIFAR10(
            root="data/raw",
            download=True,
            train=False,
            transform=self.target_transforms,
        )

        return DataLoader(
            dataset,
            batch_size=64,
            num_workers=4,
            shuffle=True,
        )

    # TODO -> Implement testing_step()
