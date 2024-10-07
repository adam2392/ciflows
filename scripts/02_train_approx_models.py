import argparse
import logging
import os
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from normflows.flows.affine import GlowBlock
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ciflows.glow import InjectiveGlowBlock, Squeeze
from ciflows.lightning import plApproximateFlowModel
from ciflows.vit import VisionTransformerDecoder, VisionTransformerEncoder


def get_encoder():
    pass


def get_decoder():
    pass


def get_latent_distr():
    pass


def get_model():

    input_shape = (1, 28, 28)
    img_size = input_shape[1]
    in_channels = input_shape[0]
    patch_size = 4
    embed_dim = 64

    n_heads = 4
    hidden_dim = 1024
    n_layers = 3

    # output shape
    n_patches = (img_size // patch_size) ** 2
    output_shape = (64,)

    debug = True

    # add the initial mixing layers
    print("Beginning of setting up transformer models...")
    # Instantiate the encoder and decoder
    encoder = VisionTransformerEncoder(
        img_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, n_layers
    )
    decoder = VisionTransformerDecoder(
        img_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, n_layers
    )
    q0 = nf.distributions.DiagGaussian(output_shape)
    return q0, encoder, decoder


def initialize_model(model):
    """
    Initialize a full normalizing flow model
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # Layer-dependent initialization
            if "coupling" in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            else:
                try:
                    nn.init.xavier_uniform_(param)
                except ValueError:
                    nn.init.normal_(param, mean=0.0, std=0.01)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (-0.5,))]
        )

    def setup(self, stage: str):
        self.mnist_test = MNIST(
            self.data_dir, download=True, train=False, transform=self.transform
        )
        mnist_full = MNIST(
            self.data_dir, download=True, train=True, transform=self.transform
        )
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # log_dir = args.log_dir
    # log_dir = '/home/adam2392/projects/logs/'
    seed = 1234

    # set seed
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")
    debug = False
    fast_dev = False
    if debug:
        accelerator = "cpu"
        fast_dev = True

    batch_size = 256
    max_epochs = 1000
    devices = 1
    strategy = "auto"  # or ddp if distributed
    num_workers = 4
    gradient_clip_val = None  # 1.0
    check_val_every_n_epoch = 5
    monitor = "val_loss"

    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"
    hutchinson_samples = 5

    # whether or not to shuffle dataset
    shuffle = True

    # output filename for the results
    root = "./data/"
    model_name = "check_fff_mnist_v1"
    checkpoint_dir = Path("./results") / model_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model_fname = f"{model_name}-model.pt"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor=monitor,
        every_n_epochs=check_val_every_n_epoch,
    )

    # logger = TensorBoardLogger(
    #     "check_injflow_mnist_logs",
    #     name="check_injflow_mnist",
    #     version="01",
    #     log_graph=True,
    # )
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n\n\tsaving to {model_fname} \n")

    # Define the trainer
    trainer = pl.Trainer(
        logger=True,
        max_epochs=max_epochs,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        fast_dev_run=fast_dev,
    )

    # epoch=99
    # step=43000
    # model_fname = checkpoint_dir / f'epoch={epoch}-step={step}.ckpt'
    # model = plFlowModel.load_from_checkpoint(model_fname)

    # define the model
    q0, encoder, decoder = get_model()
    initialize_model(encoder)
    initialize_model(decoder)
    model = plApproximateFlowModel(
        latent=q0,
        encoder=encoder,
        decoder=decoder,
        lr=lr,
        lr_min=lr_min,
        lr_scheduler=lr_scheduler,
        hutchinson_samples=hutchinson_samples,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # define the data loader
    data_module = MNISTDataModule(
        data_dir=root,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    # save the final model
    print(f"Saving model to {checkpoint_dir / 'final_model.pt'}")
    torch.save(model, checkpoint_dir / f"{model_fname}_final.pt")
