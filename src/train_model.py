import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchvision.transforms import v2

from models.ImageNet import ImageNet


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):
    """Function to train the model"""

    print(f"Train modeling using {config.data.raw}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {config.data.final}")

    # Data augmentation and normalization for training
    # Just normalization for validation
    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomRotation(degrees=(-30, 30)),
            v2.RandomResizedCrop(
                size=(config.model.aug_height, config.model.aug_width)
            ),
            v2.RandomHorizontalFlip(p=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(config.model.img_height * 3, config.model.img_width * 3)),
            v2.CenterCrop(size=(config.model.aug_height, config.model.aug_width)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = ImageNet(
        config.model.num_classes,
        config.model.batch_size,
        train_transforms,
        test_transforms,
    )
    # logger = CSVLogger(save_dir="logs", name="csv")
    logger = TensorBoardLogger(save_dir="logs", name="image_net_v1")
    early_stopping = EarlyStopping(monitor="train_loss", patience=5)

    trainer = Trainer(
        max_epochs=config.model.epochs, logger=logger, callbacks=[early_stopping]
    )

    trainer.fit(model)


if __name__ == "__main__":
    train_model()
