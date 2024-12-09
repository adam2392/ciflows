import argparse
import logging
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ciflows.datasets.celeba_scm.pretrained import (
    CelebaDataModule,
    MultiTaskResNet,
    PredictorPipeline,
)

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # Initialize Data Module
    root_dir = Path("/Users/adam2392/pytorch_data/")  # Specify the path to the dataset
    batch_size = 32
    max_epochs = 100
    accelerator = "mps"
    num_workers = 4  # Adjust based on your system capabilities

    # Create directory to save checkpoints
    checkpoints_dir = root_dir / "celeba" / "predictor-64x64" / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    data_module = CelebaDataModule(
        root=root_dir, batch_size=batch_size, num_workers=num_workers
    )

    # Initialize Model
    base_model = MultiTaskResNet()
    model = PredictorPipeline(model=base_model, lr=1e-3)

    #  Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="checkpoint-epoch{epoch:02d}",  # Save with epoch in filename
        save_top_k=-1,  # Save all checkpoints
        save_on_train_epoch_end=True,  # Save at the end of each epoch
        every_n_epochs=5,  # Save every 5 epochs
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,  # Use "cpu" if GPU is not available
        devices=1,  # Number of GPUs, use `auto` for all available GPUs
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
        log_every_n_steps=10,
    )

    # Train the Model
    trainer.fit(model, data_module)

    # Save Final Model
    final_model_path = "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")

    # Validate the Model (optional, since validation happens during training)
    trainer.validate(model, data_module)
    logging.info("Succesfully completed :)\n\n")
