import logging
import math
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from normflows.flows.affine import GlowBlock
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ciflows.flows import TwoStageTraining, plInjFlowModel
from ciflows.flows.glow import InjectiveGlowBlock, Squeeze


def get_inj_model():
    use_lu = True
    gamma = 1e-6
    activation = "linear"

    n_hidden = 512
    n_glow_blocks = 3
    n_mixing_layers = 2
    n_injective_layers = 4
    n_layers = n_mixing_layers + n_injective_layers

    input_shape = (1, 32, 32)
    n_channels = input_shape[0]
    img_size = input_shape[1]

    n_chs = n_channels
    flows = []

    debug = False

    n_chs = int(n_channels * 4**n_mixing_layers * (1 / 2) ** n_injective_layers)
    print("Starting at latent representation: ", n_chs)
    latent_size = int(img_size / (2**n_mixing_layers))
    q0 = nf.distributions.DiagGaussian(
        (n_chs, latent_size, latent_size), trainable=False
    )

    split_mode = "channel"

    for i in range(n_injective_layers):
        if i == 0:
            split_mode = "checkerboard"
        else:
            split_mode = "channel"

        for j in range(n_glow_blocks):
            flows += [
                GlowBlock(
                    channels=n_chs,
                    hidden_channels=n_hidden,
                    use_lu=use_lu,
                    scale=True,
                    split_mode=split_mode,
                )
            ]

        # input to inj flow is what is at the X -> V layer
        flows += [
            InjectiveGlowBlock(
                channels=n_chs,
                hidden_channels=n_hidden,
                activation=activation,
                scale=True,
                gamma=gamma,
                debug=debug,
                split_mode=split_mode,
            )
        ]
        n_chs = n_chs * 2
        if debug:
            print(f"On layer {n_layers - i}, n_chs = {n_chs//2} -> {n_chs}")

    for i in range(n_mixing_layers):
        for j in range(n_glow_blocks):
            flows += [
                GlowBlock(
                    channels=n_chs,
                    hidden_channels=n_hidden,
                    use_lu=use_lu,
                    scale=True,
                    split_mode=split_mode,
                )
            ]
        flows += [Squeeze()]
        n_chs = n_chs // 4
        if debug:
            print(f"On layer {n_mixing_layers - i}, n_chs = {n_chs}")

    model = nf.NormalizingFlow(q0=q0, flows=flows)
    model.output_n_chs = n_chs
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    return model


def get_bij_model(n_chs, latent_size):
    use_lu = True
    n_hidden = 256
    n_glow_blocks = 6

    flows = []

    debug = False

    print("Starting at latent representation: ", n_chs, latent_size, latent_size)
    q0 = nf.distributions.DiagGaussian((n_chs, latent_size, latent_size), trainable=True)

    split_mode = "checkerboard"

    for i in range(n_glow_blocks):
        flows += [
            GlowBlock(
                channels=n_chs,
                hidden_channels=n_hidden,
                use_lu=use_lu,
                scale=True,
                split_mode=split_mode,
            )
        ]

        if debug:
            print(f"On layer {n_glow_blocks - i}, n_chs = {n_chs//2} -> {n_chs}")

    model = nf.NormalizingFlow(q0=q0, flows=flows)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    return model


def initialize_flow(model):
    """
    Initialize a full normalizing flow model
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # Layer-dependent initialization
            if "coupling" in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            else:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle=True,
        fast_dev_run=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.fast_dev_run = fast_dev_run

    def setup(self, stage: str):
        self.mnist_test = MNIST(
            self.data_dir, download=True, train=False, transform=self.transform
        )
        mnist_full = MNIST(
            self.data_dir, download=True, train=True, transform=self.transform
        )
        if self.fast_dev_run:
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [100, 60_000 - 100],
                generator=torch.Generator().manual_seed(42),
            )
        else:
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

    batch_size = 1024
    devices = 1
    strategy = "auto"  # or ddp if distributed
    num_workers = 6
    gradient_clip_val = 1.0
    check_val_every_n_epoch = 1
    monitor = "val_loss"

    n_steps_mse = 20
    mse_chkpoint_name = f"mse_chkpoint_{n_steps_mse}"

    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"

    # whether or not to shuffle dataset
    shuffle = True

    # output filename for the results
    root = "./data/"
    model_name = "injflow_twostage_batch1024_gradclip1_mnist_trainableq0_nstepsmse20_v2"
    checkpoint_dir = Path("./results") / model_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # epoch=99
    # step=43000
    # model_fname = checkpoint_dir / f'epoch={epoch}-step={step}.ckpt'
    # model = plFlowModel.load_from_checkpoint(model_fname)

    # define the model
    inj_model = get_inj_model()
    samples = inj_model.q0.sample(2)
    _, n_chs, latent_size, _ = samples.shape
    print(samples.shape)
    bij_model = get_bij_model(n_chs=n_chs, latent_size=latent_size)

    initialize_flow(inj_model)
    initialize_flow(bij_model)

    debug = False
    fast_dev = False
    max_epochs = 500
    if debug:
        accelerator = "cpu"
        fast_dev = True
        max_epochs = 1
        batch_size = 2
    # else:
        # torch.set_float32_matmul_precision("high")
        # model = torch.compile(model)

    model = plInjFlowModel(
        inj_model=inj_model,
        bij_model=bij_model,
        lr=lr,
        lr_min=lr_min,
        lr_scheduler=lr_scheduler,
        n_steps_mse=n_steps_mse,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=mse_chkpoint_name,
        debug=debug,
        check_val_every_n_epoch=check_val_every_n_epoch
    )

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
    # logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print()
    print(f"Model name: {model_name}")
    print()

    # configure logging at the root level of Lightning
    # logging.getLogger("lightning.pytorch").setLevel(level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    # Define the trainer
    trainer = pl.Trainer(
        logger=False,
        max_epochs=max_epochs,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback, TwoStageTraining()],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        gradient_clip_val=gradient_clip_val,
        # fast_dev_run=fast_dev,
        # log_every_n_steps=1,
        # max_epochs=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    # define the data loader
    data_module = MNISTDataModule(
        data_dir=root,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # fast_dev_run=fast_dev,
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    # save the final model
    print(f"Saving model to {checkpoint_dir / '{model_name}_final.pt'}")
    torch.save(model, checkpoint_dir / f"{model_name}_final.pt")
