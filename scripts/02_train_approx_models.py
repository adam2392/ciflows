import logging
from collections import OrderedDict
from math import prod
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from fff.model.utils import wrap_batch_norm2d
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ciflows.lightning import plApproximateFlowModel
from ciflows.vit import VisionTransformerDecoder, VisionTransformerEncoder


class SkipConnection(nn.Module):
    def __init__(self, inner: nn.Module, id_init=False):
        super().__init__()
        self.inner = inner
        if id_init:
            self.scale = torch.nn.Parameter(torch.zeros(1))
        else:
            self.scale = None

    def forward(self, x):
        out = self.inner(x)
        if self.scale is not None:
            out = out * self.scale
        return x[..., : out.shape[-1]] + out


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_dim=128,
        latent_dim=128,
        skip_connection=False,
        ch_factor=128,
        encoder_spec=None,
        decoder_spec=None,
        batch_norm=False,
        instance_norm=False,
    ):
        super().__init__()

        # Default specifications for encoder and decoder if not provided
        # out_channels_factor, kernel_size, stride, padding
        self.encoder_spec = encoder_spec or [
            [1, 4, 2, 1],
            [2, 4, 2, 1],
            [4, 4, 2, 1],
            [8, 4, 2, 1],
        ]
        self.decoder_spec = decoder_spec or [
            [4, 3, 2, 1],
            [2, 3, 2, 1, 1],
            [1, 3, 2, 1, 1],
        ]
        self.ch_multiplier = 8
        self.spatial_dim = 4

        # Set other parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.skip_connection = skip_connection
        self.ch_factor = ch_factor
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm

        # Build the model with encoder and decoder modules
        self.model = self.build_model()

    def build_model(self):
        encoder = self.build_encoder()
        decoder = self.build_decoder()

        # Apply skip connections if specified
        modules = OrderedDict(encoder=encoder, decoder=decoder)
        if self.skip_connection:
            modules = OrderedDict((k, SkipConnection(v)) for k, v in modules.items())
        return nn.Sequential(modules)

    def build_encoder(self):
        image_shape = (self.in_channels, 32, 32)
        encoder = nn.Sequential()

        n_channels = self.in_channels
        # add layers to encoder
        for i, conv_spec in enumerate(self.encoder_spec):
            out_channels, kernel_size, stride, padding = conv_spec
            out_channels *= self.ch_factor
            encoder.append(
                nn.Conv2d(
                    n_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            # if self.batch_norm:
            #     encoder.append(wrap_batch_norm2d(self.batch_norm, out_channels))
            if self.instance_norm:
                encoder.append(nn.InstanceNorm2d(out_channels))
            encoder.append(nn.ReLU())
            n_channels = out_channels

        # flattens (batch, channels, height, width) to (batch, channels * height * width)
        encoder.append(nn.Flatten(-3, -1))
        # project to latent space
        encoder.append(nn.Linear(n_channels, self.latent_dim))
        return encoder

    def build_decoder(self):
        decoder = nn.Sequential()
        latent_channels, latent_size = self.decoder_spec[0]
        latent_channels *= self.ch_factor
        n_channels = latent_channels + self.hidden_dim

        decoder.append(
            nn.Linear(self.latent_dim + self.hidden_dim, n_channels * prod(latent_size))
        )
        decoder.append(nn.Unflatten(-1, (n_channels, *latent_size)))

        for i, conv_spec in enumerate(self.decoder_spec[1:]):
            is_last_layer = i + 1 == len(self.decoder_spec) - 1
            out_channels, *args = conv_spec
            out_channels = (
                out_channels * self.ch_factor
                if not is_last_layer
                else self.out_channels
            )
            decoder.append(nn.ConvTranspose2d(n_channels, out_channels, *args))
            if not is_last_layer:
                if self.batch_norm:
                    decoder.append(wrap_batch_norm2d(self.batch_norm, out_channels))
                if self.instance_norm:
                    decoder.append(nn.InstanceNorm2d(out_channels))
                decoder.append(nn.ReLU())
            else:
                decoder.append(nn.Sigmoid())
            n_channels = out_channels
        decoder.append(nn.Flatten(-3, -1))
        return decoder


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

    batch_size = 256
    devices = 1
    strategy = "auto"  # or ddp if distributed
    num_workers = 6
    gradient_clip_val = 1.0
    check_val_every_n_epoch = 1
    check_samples_every_n_epoch = 5
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
    train_from_checkpoint = False

    model_fname = None
    # define the model
    inj_model = get_inj_model()
    samples = inj_model.q0.sample(2)
    _, n_chs, latent_size, _ = samples.shape
    print(samples.shape)
    initialize_flow(inj_model)

    # bij_model = None
    bij_model = get_bij_model(n_chs=n_chs, latent_size=latent_size)
    initialize_flow(bij_model)

    debug = False
    fast_dev = False
    max_epochs = 2000
    if debug:
        accelerator = "cpu"
        fast_dev = True
        max_epochs = 5
        n_steps_mse = 1
        batch_size = 16
        check_samples_every_n_epoch = 1
    else:
        torch.set_float32_matmul_precision("high")

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
        gradient_clip_val=gradient_clip_val,
    )

    # epoch=99
    # step=43000
    # model_fname = checkpoint_dir / f'epoch={epoch}-step={step}.ckpt'
    # model = plFlowModel.load_from_checkpoint(model_fname)

    # define the data loader
    data_module = MNISTDataModule(
        data_dir=root,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        fast_dev_run=fast_dev,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path=model_fname
    )

    # save the final model
    print(f"Saving model to {checkpoint_dir / '{model_name}_final.pt'}")
    torch.save(model, checkpoint_dir / f"{model_fname}_final.pt")
