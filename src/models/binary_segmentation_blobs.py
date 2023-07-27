import os
from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from segmentation_models_pytorch import create_model
from torch.distributions.bernoulli import Bernoulli
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification.f_beta import BinaryF1Score
from torchmetrics.classification.precision_recall import (BinaryPrecision,
                                                          BinaryRecall)
from torchvision.utils import make_grid, save_image
from transformers.optimization import get_cosine_schedule_with_warmup


class BinarySegmentationReinforceModel(pl.LightningModule):
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
        warmup_epochs: int = 0,
    ):
        """Binary Segmentation Model with Reinforce Loss

        Args:
            arch: Name of segmentation model architecture
            encoder: Name of encoder for segmentation model
            optimizer: Name of optimizer (adam | adamw | sgd)
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler (cosine | onecycle | none)
            warmup_epochs: Number of warmup epochs
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
        self.warmup_epochs = warmup_epochs

        # Initialize network
        self.net = create_model(
            self.arch,
            encoder_name=self.encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        # Metrics
        self.train_metrics = MetricCollection([BinaryF1Score()])
        self.val_metrics = MetricCollection(
            [BinaryF1Score(), BinaryPrecision(), BinaryRecall()]
        )
        self.test_metrics = MetricCollection(
            [BinaryF1Score(), BinaryPrecision(), BinaryRecall()]
        )

        # Initialize blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 10000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def forward(self, x):
        return self.net(x)

    def log_samples(self, inp, pred, target) -> None:
        """Log sample outputs"""
        """
        redo as callback: https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/callbacks/vision/image_generation.py
        """
        # Only log up to 16 images
        inp, pred, target = inp[:16], pred[:16], target[:16]

        # Log result
        if "CSVLogger" in str(self.logger.__class__):
            path = os.path.join(self.logger.log_dir, "samples")  # type:ignore
            if not os.path.exists(path):
                os.makedirs(path)
            filename = os.path.join(path, str(self.current_epoch) + "ep.png")
            save_image(pred.unsqueeze(1), filename, nrow=4)
            filename = os.path.join(path, str(self.current_epoch) + "ep_gt.png")
            save_image(target.unsqueeze(1), filename, nrow=4)
        elif "WandbLogger" in str(self.logger.__class__):
            grid = make_grid(pred.unqueuze(1), nrow=4)
            self.logger.log_image(key="sample", images=[grid])  # type:ignore

    def count_reward(self, batch, n):
        # Count the number of predicted blobs
        n_pred = []
        for sample in batch:
            sample = sample.detach().cpu().numpy() * 255
            sample = np.uint8(sample)
            _, sample = cv2.threshold(sample, 127, 255, cv2.THRESH_BINARY_INV)
            sample = cv2.copyMakeBorder(
                sample, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            n_pred.append(len(self.blob_detector.detect(sample)))

        print(n_pred)
        print(n)
        exit()

    def shared_step(self, batch, idx=None, mode="train"):
        x, y, n = batch

        # Pass through network
        pred = self.net(x).squeeze(1)

        # Calculate MLE loss
        loss = F.binary_cross_entropy_with_logits(pred, y)

        # Generate sample and baseline by sampling from predicted probs
        dist = Bernoulli(logits=pred)
        sample = dist.sample()
        baseline = dist.sample()

        reward = self.count_reward(y, n)

        # Get metrics
        metrics = getattr(self, f"{mode}_metrics")(pred, y)

        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            self.log(f"{mode}_{k.lower()}", v, on_epoch=True)
        if mode == "val" and idx == 0:
            self.log_samples(x, pred, y)

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        """Setup optimizer and scheduler"""

        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            epoch_steps = (
                self.trainer.estimated_stepping_batches
                // self.trainer.max_epochs  # type:ignore
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,  # type:ignore
                num_warmup_steps=epoch_steps * self.warmup_epochs,  # type:ignore
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
