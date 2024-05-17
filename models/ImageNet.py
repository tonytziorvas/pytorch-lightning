import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import SGD, Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10

from models.ConvBlock import ConvBlock
from models.FcBlock import FcBlock


class ImageNet(pl.LightningModule):
    def __init__(self, num_classes, batch_size, transforms, test_transforms, optim, lr):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.transforms = transforms
        self.test_transforms = test_transforms
        self.optim = optim
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # self.save_hyperparameters(logger=False, ignore=["transforms"])

        layers = [
            ConvBlock(3, 16, kernel_size=3, padding=1, bias=False),
            ConvBlock(16, 32, kernel_size=3, padding=1, bias=False),
            ConvBlock(32, 64, kernel_size=3, padding=1, bias=False),
            ConvBlock(64, 128, kernel_size=3, padding=1, bias=False),
            nn.Flatten(start_dim=1),
            FcBlock(128 * 4 * 4, 128),
            FcBlock(128, 64, use_norm=True),
            FcBlock(64, num_classes),
        ]

        self.net = nn.Sequential(*layers)

    def train_val_split(self):
        pass

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

    def test_step(self, batch, batch_idx):
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

    def configure_optimizers(self, **kwargs) -> Optimizer:
        """
        Configure the optimizer for the model.

        Args:
            optimizer (str, optional): The optimizer to use. Defaults to "adam".
            **kwargs: Additional keyword arguments to be passed to the optimizer module.

        Returns:
            torch.optim.Optimizer: The optimizer instance
        """
        optimizer_map = {
            "adam": Adam,
            "sgd": SGD,
            "rmsprop": RMSprop,
        }

        if self.optim not in optimizer_map:
            raise ValueError(f"Invalid optimizer: {self.optim}")

        return optimizer_map[self.optim](self.net.parameters(), lr=self.lr, **kwargs)

    def train_dataloader(self) -> DataLoader:
        # TODO Move Dataset in a separate class
        dataset = CIFAR10(
            root="data/raw",
            download=True,
            train=True,
            transform=self.transforms,
        )

        return DataLoader(
            dataset,
            self.batch_size,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = CIFAR10(
            root="data/raw",
            download=True,
            train=False,
            transform=self.test_transforms,
        )

        return DataLoader(
            dataset,
            self.batch_size,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
        )
