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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
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
        # loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x)
        loss_reconstruction = ((x - x_hat) ** 2).reshape(B, -1).sum(-1)

        # print(x.shape, x_hat.shape)
        # assert torch.allclose(x.shape, x_hat.shape, atol=1e-5)
        # get negative log likelihoood
        v_hat = v_hat.view(B, -1)
        loss_nll = -self.latent.log_prob(v_hat) - surrogate_loss

        # print(loss_reconstruction.shape, loss_nll.shape)
        loss = (self.beta * loss_reconstruction + loss_nll).mean()

        if batch_idx % 100 == 0:
            print()
            print(f"train_loss: {loss.item():.3f} | recon_loss: {loss_reconstruction.mean().item():.3f} | nll_loss: {loss_nll.mean().item():.3f} | surrogate_loss: {surrogate_loss.mean().item():.3f}")
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

        loss = (self.beta * loss_reconstruction + loss_nll).mean()
        # Print the loss to the console
        if batch_idx % 100 == 0:
            print(f"val_loss: {loss.item():.3f}")
        self.log("val_loss", loss)
        return loss


if __name__ == "__main__":
    # Set up training configurations
    batch_size = 1024
    num_workers = 8
    shuffle = True

    load_from_checkpoint = True
    # latentdim128-beta5
    # latent_dim = 128
    # epoch = 209
    # step = 45150
    # model_name = "check_fif_convvae_mnist_latentdim128_beta5_v2"
    # latentdim12-beta5
    latent_dim = 64
    epoch = 24
    step=1350
    # model_name = "check_fif_convvae_mnist_latentdim12_beta5_v3"
    model_name = "check_fif_convvae_mnist_batch1024_latentdim64_beta100_v1"

    beta = 100

    max_epochs = 1000
    hutchinson_samples = 1
    gradient_clip_val = 1.0
    lr = 3e-4
    lr_min = 1e-7
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
    encoder = ConvNetEncoder(latent_dim=latent_dim, in_channels=channels, hidden_dim=1024, start_channels=32*4, debug=False)
    decoder = ConvNetDecoder(latent_dim=latent_dim, out_channels=channels, hidden_dim=1024, start_channels=32*4, debug=False)
    latent = DiagGaussian(latent_dim)

    if load_from_checkpoint:
        # XXX: Unsure why the loading from checkpoint does not work?
        # Load the model from a checkpoint
        checkpoint_path = checkpoint_dir / f"epoch={epoch}-step={step}.ckpt"
        model = plFFFConvVAE.load_from_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        current_max_epochs = checkpoint["epoch"]
        max_epochs += current_max_epochs
    else:
        checkpoint_path = None
        model = plFFFConvVAE(
            encoder,
            decoder,
            latent,
            lr=lr,
            lr_min=lr_min,
            lr_scheduler=lr_scheduler,
            hutchinson_samples=hutchinson_samples,
            beta=beta,
        )

    debug = False
    fast_dev = False
    if debug:
        accelerator = "cpu"
        fast_dev = True
        max_epochs = 1
        batch_size = 2
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
        fast_dev_run=fast_dev,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=checkpoint_path
    )

    # save the final model
    print(f"Saving model to {checkpoint_dir / 'final_model.pt'}")
    torch.save(model, checkpoint_dir / f"{model_fname}_final.pt")
