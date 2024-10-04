import argparse
import os
from pathlib import Path

import torch.nn as nn
import lightning as pl
import normflows as nf
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from normflows.flows.affine import GlowBlock
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ciflows.glow import InjectiveGlowBlock, Squeeze
from ciflows.lightning import plFlowModel


def get_model():
    n_hidden = 64
    n_mixing_layers = 2
    n_injective_layers = 3
    n_glow_blocks = 2
    use_lu = True
    gamma = 1e-2
    activation = "linear"

    input_shape = (1, 28, 28)
    n_channels = input_shape[0]

    n_chs = n_channels
    flows = []

    debug = True

    # add the initial mixing layers
    print("Beginning of mixing flows.")
    mixing_flows = []
    # Add flow layers starting from the latent representation
    for i in range(n_mixing_layers):
        # n_chs = C * 4^(L - i)
        n_chs = n_channels * 4 ** (n_mixing_layers - i)

        if debug:
            print(f"On layer {n_mixing_layers - i}, n_chs = {n_chs}")
        for j in range(n_glow_blocks):
            mixing_flows += [
                GlowBlock(
                    channels=n_chs,
                    hidden_channels=n_hidden,
                    use_lu=use_lu,
                    scale=True,
                )
            ]
        mixing_flows += [Squeeze()]

    # reverse the mixing flows to go from X -> V.
    mixing_flows = mixing_flows[::-1]
    i = 1
    for flow in mixing_flows:
        if hasattr(flow, "n_channels"):
            print(f"On layer {i}, n_chs = {flow.n_channels}")
            i += 1
    num_layers = i

    print("Beginning of injective flows.")
    n_chs = n_channels * 4 ** (n_mixing_layers - 0)
    debug = True
    # add injective blocks
    injective_flows = []
    for i in range(n_injective_layers):
        # Note: this is adding from V -> X
        # n_chs = n_chs // 2
        # injective_flows += [
        #     InjectiveGlowBlock(
        #         channels=n_chs,
        #         hidden_channels=n_hidden,
        #         activation=activation,
        #         scale=True,
        #         gamma=gamma,
        #     )
        # ]

        if debug:
            print(f"On layer {i + num_layers}, n_chs = {n_chs}")
        for j in range(n_glow_blocks):
            injective_flows += [
                GlowBlock(
                    channels=n_chs,
                    hidden_channels=n_hidden,
                    use_lu=use_lu,
                    scale=True,
                )
            ]

    # Note: this is constructed as X -> V, so we need to reverse the flows
    # to adhere to the normflows convention of V -> X
    flows = mixing_flows
    flows.extend(injective_flows)
    flows = flows[::-1]

    print("n_channels: ", n_chs)
    q0 = nf.distributions.DiagGaussian((n_chs, 7, 7))

    model = nf.NormalizingFlow(q0=q0, flows=flows)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    return model


def initialize_flow(model):
    """
    Initialize a full normalizing flow model
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Layer-dependent initialization
            if 'coupling' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            else:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
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
        self.mnist_test = MNIST(self.data_dir, download=True, train=False, transform=self.transform)
        mnist_full = MNIST(self.data_dir, download=True, train=True, transform=self.transform)
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

    batch_size = 128
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

    # whether or not to shuffle dataset
    shuffle = True

    # output filename for the results
    root = "./data/"
    model_name = "check_glowflow_mnist_v1"
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
    logger = None

    # Define the trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        # fast_dev_run=True,
    )

    # define the model
    flow_model = get_model()
    initialize_flow(flow_model)
    model = plFlowModel(flow_model, lr=lr, lr_min=lr_min, lr_scheduler=lr_scheduler)

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
