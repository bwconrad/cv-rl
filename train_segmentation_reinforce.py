from typing import Optional, Tuple

import torch
from jsonargparse import lazy_instance
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger
from segmentation_models_pytorch import create_model
from torch.distributions.bernoulli import Bernoulli
from torchmetrics import Dice, MetricCollection
from torchmetrics.classification.f_beta import BinaryF1Score
from torchmetrics.classification.precision_recall import (BinaryPrecision,
                                                          BinaryRecall)
from torchmetrics.functional.classification.f_beta import binary_f1_score

from src.base_model import BaseModel
from src.callbacks import SegmentationImageGridSampler
from src.datamodules import SegmentationDataModule


class BinaryF1SegmentationModel(BaseModel):
    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet18",
        optimizer: str = "adam",
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "none",
        warmup_steps: int = 0,
        weights: Optional[str] = None,
        prefix: str = "net",
    ):
        """MLE Binary Segmentation Model

        Args:
            arch: Name of segmentation model architecture
            encoder: Name of encoder for segmentation model
            optimizer: Name of optimizer (adam | adamw | sgd)
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler (cosine | onecycle | none)
            warmup_steps: Number of warmup epochs
            weights: Path to model weights
            prefix: Parameter prefix to strip when loading weights
        """
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.weights = weights
        self.prefix = prefix

        # Initialize network
        self.net = create_model(
            self.arch,
            encoder_name=self.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        # Load weights
        if self.weights:
            state_dict = torch.load(self.weights)
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith(self.prefix):
                    continue

                k = k.replace(self.prefix + ".", "")
                new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)

        # Metrics
        self.train_metrics = MetricCollection([BinaryF1Score()])
        self.val_metrics = MetricCollection(
            [BinaryF1Score(), BinaryPrecision(), BinaryRecall(), Dice()]
        )
        self.test_metrics = MetricCollection(
            [BinaryF1Score(), BinaryPrecision(), BinaryRecall()]
        )

    def shared_step(self, batch, mode="train"):
        x, y = batch

        # Pass through network
        pred = self.net(x).squeeze(1)

        # Calculate MLE loss
        # loss = F.binary_cross_entropy_with_logits(pred, y)

        # Generate sample and baseline by sampling from predicted probs
        dist = Bernoulli(logits=pred)
        sample = dist.sample()
        baseline = dist.sample()

        # Calculate reward (F1) with baseline
        reward = binary_f1_score(sample, y) - binary_f1_score(baseline, y)

        # Get the log prob of sampling each prediction in batch
        log_prob = torch.mean(dist.log_prob(sample), dim=[1, 2])

        # Calculate reinforce loss
        loss = torch.mean(-log_prob * reward)

        # Get metrics
        metrics = getattr(self, f"{mode}_metrics")(pred, y.long())

        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir="output", name="default"
                )
            }
        )
        parser.add_lightning_class_args(SegmentationImageGridSampler, "image_sampler")


if __name__ == "__main__":
    MyLightningCLI(
        BinaryF1SegmentationModel,
        SegmentationDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={"check_val_every_n_epoch": None},
    )
