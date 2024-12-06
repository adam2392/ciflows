import os
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import random_split

from ciflows.datasets.causalceleba import CausalCelebA
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.reduction.vae import VAE
from ciflows.training import TopKModelSaver
from ciflows.eval import load_model


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): File path to save the best model.
            verbose (bool): If True, prints a message when an improvement occurs.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss  # Negative because lower validation loss is better.

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def data_loader(
    root_dir,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
    val_split=0.2,
    img_size=64,
):
    # Define the image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size), antialias=True
            ),  # Resize images to 128x128
            transforms.CenterCrop(img_size),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    causal_celeba_dataset = CausalCelebA(
        root=root_dir,
        graph_type=graph_type,
        transform=image_transform,
        fast_dev_run=False,  # Set to True for debugging
    )

    # Calculate the number of samples for training and validation
    total_len = len(causal_celeba_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(
        causal_celeba_dataset, [train_len, val_len]
    )

    distr_labels = [x[1][-1] for x in causal_celeba_dataset]
    unique_distrs = len(np.unique(distr_labels))
    if batch_size < unique_distrs:
        raise ValueError(
            f"Batch size must be at least {unique_distrs} for stratified sampling."
        )
    train_sampler = StratifiedSampler(distr_labels, batch_size)

    # Define the DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        # shuffle=True,  # Shuffle data during training
        num_workers=num_workers,
        pin_memory=True,  # Enable if using a GPU
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle data during validation
        num_workers=num_workers // 2,
        pin_memory=True,  # Enable if using a GPU
    )

    return train_loader, val_loader


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var, image_dim):
    MSE = F.mse_loss(recon_x, x.view(-1, image_dim))
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD
    return loss


if __name__ == "__main__":
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

    root = Path("/Users/adam2392/pytorch_data/celeba")
    root = Path("/home/adam2392/projects/data/")


    latent_dim = 48
    batch_size = 256
    model_fname = "celeba_vaereduction_batch256_latentdim48_img128_v1.pt"
    
    checkpoint_dir = root / "CausalCelebA" / "vae_reduction" / model_fname.split(".")[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = 1000
    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"
    debug = False
    num_workers = 4
    graph_type = "chain"

    torch.set_float32_matmul_precision("high")
    if debug:
        accelerator = "cpu"
        # device = 'cpu'
        max_epochs = 5
        batch_size = 16
        check_samples_every_n_epoch = 1
        num_workers = 2

        fast_dev = True

    model = VAE(LATENT_DIM=latent_dim)
    model = model.to(device)
    img_size = 128
    image_dim = 3 * img_size * img_size

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Cosine Annealing Scheduler (adjust the T_max for the number of epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )  # T_max = total epochs

    top_k_saver = TopKModelSaver(
        checkpoint_dir, k=5
    )  # Initialize the top-k model saver

    train_loader, val_loader = data_loader(
        root_dir=root,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
        img_size=img_size,
    )

    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # training loop
    # - log the train and val loss every 10 epochs
    # - sample from the model every 10 epochs, and save the images
    # - save the top 5 models based on the validation loss
    # - save the model at the end of training

    # Training loop
    for epoch in tqdm(range(1, max_epochs + 1), desc="outer", position=0):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, meta_labels, targets) in tqdm(
            enumerate(train_loader), desc="step", position=1, leave=False
        ):
            torch.cuda.empty_cache()
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, latent_mu, latent_logvar = model(
                images
            )  # Model forward pass

            # Clamp logvar to prevent numerical instability
            latent_logvar = torch.clamp_(latent_logvar, -10, 10)

            loss = loss_function(
                reconstructed, images, latent_mu, latent_logvar, image_dim=image_dim
            )  # Custom VAE loss function
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Step the scheduler at the end of the epoch
        scheduler.step()

        train_loss /= len(train_loader)
        print(
            f"====> Epoch: {epoch} Average Train loss: {train_loss:.4f}, LR: {lr:.6f}"
        )

        # Validation phase
        model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for val_images, target in val_loader:
                reconstructed, latent_mu, latent_logvar = model(
                    val_images
                )  # Model forward pass

                loss = loss_function(
                    reconstructed,
                    val_images,
                    latent_mu,
                    latent_logvar,
                    image_dim=image_dim,
                )  # Custom VAE loss function
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"====> Epoch: {epoch} Average Val loss: {val_loss:.4f}")

        # Log training and validation loss
        if debug or epoch % 10 == 0:
            print()
            print(
                f"Saving images - Epoch [{epoch}/{max_epochs}], Train Loss: {train_loss:.4f}"
            )

            # Sample and save reconstructed images
            sample_images = images[:8]  # Pick 8 images for sampling
            reconstructed_images = model.decode(model.encode(sample_images)[0]).reshape(
                -1, 3, 64, 64
            )
            save_image(
                reconstructed_images.cpu(),
                checkpoint_dir / f"epoch_{epoch}_samples.png",
                nrow=4,
                normalize=True,
            )

        # Track top 5 models based on validation loss
        if epoch % 5 == 0:
            # Optionally, remove worse models if there are more than k saved models
            top_k_saver.save_model(model, epoch, val_loss)

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / model_fname)
    print(f"Training complete. Models saved in {checkpoint_dir}.")

    # Usage example:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = VAE().to(device)
    model_path = checkpoint_dir / model_fname
    vae_model = load_model(vae_model, model_path, device)
