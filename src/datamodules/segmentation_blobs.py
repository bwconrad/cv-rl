import os
from glob import glob
from typing import Optional, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data as data
import torchvision
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints
from torchvision.transforms.v2 import (Compose, Normalize, RandomResizedCrop,
                                       Resize)


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        size: int = 256,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        batch_size: int = 32,
        workers: int = 4,
    ):
        """Segmentation Datamodule

        Args:
            root: Path to root with train, val, test directories
            size: Input image size
            mean: Normalization mean
            std: Normalization standard deviation
            min_scale: Minimum crop scale
            max_scale: Maximum crop scale
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """

        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers

        # Augmentations
        self.transforms_train = Compose(
            [
                RandomResizedCrop(size, (min_scale, max_scale), antialias=True),
                Normalize(mean, std),
            ],
        )
        self.transforms_test = Compose(
            [Resize((size, size), antialias=True), Normalize(mean, std)]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = ImageMaskDataset(
                os.path.join(self.root, "train"), transforms=self.transforms_train
            )
            self.val_dataset = ImageMaskDataset(
                os.path.join(self.root, "val"), transforms=self.transforms_test
            )
        elif stage == "test":
            self.test_dataset = ImageMaskDataset(
                os.path.join(self.root, "test"), transforms=self.transforms_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


class ImageMaskDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Compose] = None,
    ):
        """Segmentation dataset with image masks

        Args:
            root: Path to dataset directory
            transforms: Data augmentations pipeline
        """
        super().__init__()

        self.transforms = transforms

        # Get image and mask file paths
        self.img_paths = sorted(
            [
                f
                for f in glob(f"{root}/images/**/*", recursive=True)
                if os.path.isfile(f)
            ]
        )
        self.mask_paths = sorted(
            [f for f in glob(f"{root}/masks/**/*", recursive=True) if os.path.isfile(f)]
        )

        self.n_blobs = pd.read_csv(
            os.path.join(root, "n-blobs.csv"), index_col="filename"
        )["n"]

        assert len(self.img_paths) == len(self.mask_paths) == len(self.n_blobs)

        print(f"Loaded {len(self.img_paths)} images from {root}")

    def __getitem__(self, index):
        # Load the image and mask
        img = datapoints.Image(
            read_image(self.img_paths[index], mode=ImageReadMode.RGB) / 255
        )
        mask = datapoints.Mask(
            read_image(self.mask_paths[index], mode=ImageReadMode.UNCHANGED) / 255
        )

        # Apply augmentations
        if self.transforms:
            img, mask = self.transforms(img, mask)

        mask = mask.squeeze(0)

        # Get number of blobs
        n = self.n_blobs[os.path.basename(self.img_paths[index])]

        return img, mask, n

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    d = ImageMaskDataset("data/tnbc/train")
    i, m, n = d[5]
    from torchvision.utils import save_image

    save_image(m.float(), "0.png")
    print(n)
