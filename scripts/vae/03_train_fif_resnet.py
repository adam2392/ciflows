from pathlib import Path

import lightning as pl
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from normflows.distributions import DiagGaussian

from ciflows.datasets.mnist import MNISTDataModule
from ciflows.loss import volume_change_surrogate
from ciflows.resnet import ConvNetDecoder, ConvNetEncoder


class plFFFConvVAE(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        latent,
        lr=3e-4,
        lr_min: float = 1e-8,
        lr_scheduler=None,
        hutchinson_samples=2,
        beta=1.0,
    ):
        super().__init__()
        # ensure that the model is saved and can be loaded later as a checkpoint
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent

        self.lr = lr
        self.lr_min = lr_min
        self.lr_scheduler = lr_scheduler
        self.hutchinson_samples = hutchinson_samples
        self.beta = torch.Tensor([beta])

    def forward(self, x, target=None):
        """Foward pass."""
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
        elif self.lr_scheduler == "step":
            # An scheduler is optional, but can help in flows to get the last bpd improvement
            scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        else:
            scheduler = None
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        # get the surrogate loss, latent representation, and reconstructed tensor
        surrogate_loss, v_hat, x_hat = volume_change_surrogate(
            x,
            self.encoder,
            self.decoder,
            hutchinson_samples=self.hutchinson_samples,
        )
        # compute reconstruction loss
        loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x)

        # get negative log likelihoood
        v_hat = v_hat.view(B, -1)
        loss_nll = -self.latent.log_prob(v_hat).mean() - surrogate_loss

        loss = self.beta * loss_reconstruction + loss_nll

        if batch_idx % 100 == 0:
            print(f"train_loss: {loss}")
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def sample(self, num_samples=1, **params):
        """
        Sample a batch of images from the flow.
        """
        # sample latent space and reshape to (batches, 1, embed_dim)
        v = self.latent.sample(num_samples, **params)
        v = v.reshape(num_samples, 1, -1)
        return self.decoder(v)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        self.beta = self.beta.to(x.device)

        # get the surrogate loss, latent representation, and reconstructed tensor
        surrogate_loss, v_hat, x_hat = volume_change_surrogate(
            x,
            self.encoder,
            self.decoder,
            hutchinson_samples=self.hutchinson_samples,
        )
        # compute reconstruction loss
        loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x).to(x.device)

        # get negative log likelihoood
        v_hat = v_hat.view(B, -1)
        loss_nll = -self.latent.log_prob(v_hat).mean() - surrogate_loss

        loss = self.beta * loss_reconstruction + loss_nll
        # Print the loss to the console
        if batch_idx % 100 == 0:
            print(f"val_loss: {loss}")
        self.log("val_loss", loss)
        return loss


if __name__ == "__main__":
    # Set up training configurations
    batch_size = 256
    num_workers = 8
    shuffle = True

    load_from_checkpoint = True
    # latentdim128-beta5
    latent_dim = 128
    epoch = 209
    step = 45150
    model_name = "check_fif_convvae_mnist_latentdim128_beta5_v2"
    # latentdim12-beta5
    # latent_dim = 12
    # epoch = 109
    # step=23650
    # model_name = "check_fif_convvae_mnist_latentdim12_beta5_v3"

    beta = 5.0

    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"
    monitor = "train_loss"
    check_val_every_n_epoch = 5
    devices = 1
    strategy = "auto"  # or ddp if distributed

    root = "./data/"
    checkpoint_dir = Path("./results") / model_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model_fname = f"{model_name}-model.pt"

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

    # Initialize your Encoder and Decoder
    channels = 1  # For grayscale images (like MNIST); set to 3 for RGB (like CelebA)
    height = 28  # Height of the input image (28 for MNIST)
    width = 28  # Width of the input image (28 for MNIST)
    encoder = ConvNetEncoder(latent_dim=latent_dim, in_channels=channels)
    decoder = ConvNetDecoder(latent_dim=latent_dim, out_channels=channels)
    latent = DiagGaussian(latent_dim)

    if load_from_checkpoint:
        # Load the model from a checkpoint
        model = plFFFConvVAE.load_from_checkpoint(checkpoint_dir / f"epoch={epoch}-step={step}.ckpt")
    else:
        model = plFFFConvVAE(
            encoder,
            decoder,
            latent,
            lr=lr,
            lr_min=lr_min,
            lr_scheduler=lr_scheduler,
            hutchinson_samples=2,
            beta=beta,
        )

    debug = False
    fast_dev = False
    max_epochs = 1000
    if debug:
        accelerator = "cpu"
        fast_dev = True
        max_epochs = 1
    else:
        torch.set_float32_matmul_precision('high')
        # model = torch.compile(model)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get the number of trainable parameters in the encoder and decoder
    num_model_params = count_trainable_parameters(model)

    # Print the results
    print(f"Total trainable parameters in Encoder: {num_model_params}")

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

    # Define the trainer
    trainer = pl.Trainer(
        logger=False,
        max_epochs=max_epochs,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
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
        fast_dev_run=fast_dev,
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    # save the final model
    print(f"Saving model to {checkpoint_dir / 'final_model.pt'}")
    torch.save(model, checkpoint_dir / f"{model_fname}_final.pt")
