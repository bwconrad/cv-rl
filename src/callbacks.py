import os
from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torchvision.utils import make_grid, save_image


class SegmentationImageGridSampler(Callback):
    def __init__(self, num_batches: int = 1, num_samples: int = 8) -> None:
        """
        Callback to save the outputs of a segmentation model.
        Saves a grid of input images, predicted masks and ground truth masks

        Args:
            num_batches: Number of batches to save
            num_samples: Number of images displayed in the grid
        """
        super().__init__()
        self.num_batches = num_batches
        self.num_samples = num_samples

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx < self.num_batches:
            x, y = batch

            # Get predicted masks
            pred = pl_module(x)
            pred = torch.round(torch.clamp(pred, 0, 1))

            # Create image grid
            inp_grid = make_grid(x[: self.num_samples], nrow=1, normalize=True)
            pred_grid = make_grid(pred[: self.num_samples], nrow=1, normalize=False)
            target_grid = make_grid(
                y[: self.num_samples].unsqueeze(1), nrow=1, normalize=False
            )
            grid = torch.cat((inp_grid, pred_grid, target_grid), -1)

            # Save according to logger
            logger = trainer.logger
            if "CSVLogger" in str(logger.__class__):
                path = os.path.join(logger.log_dir, "samples")  # type:ignore
                if not os.path.exists(path):
                    os.makedirs(path)

                filename = os.path.join(
                    path, str(trainer.global_step) + f"_{batch_idx}_ep.png"
                )
                save_image(grid, filename)
            elif "WandbLogger" in str(trainer.logger.__class__):
                logger.log_image(key="sample", images=[grid])  # type:ignore
            else:
                raise NotImplementedError(
                    f"{str(trainer.logger.__class__)} logger is not implemented for the SegmentationImageGridSampler callback"
                )
