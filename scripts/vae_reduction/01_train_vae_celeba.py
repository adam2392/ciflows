import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from ciflows.reduction.vae import VAE
import lightning as pl
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from ciflows.datasets.causalceleba import CausalCelebA
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ciflows.datasets.multidistr import StratifiedSampler


class TopKModelSaver:
    def __init__(self, save_dir, k=5):
        self.save_dir = save_dir
        self.k = k
        self.best_models = []  # List of tuples (loss, model_state_dict, epoch)

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

    def check(self, loss):
        """Determine if the current model's loss is worth saving."""
        # Check if the current model's loss should be saved
        if len(self.best_models) < self.k:
            return True  # If we have fewer than k models, always save the model
        else:
            # If the current loss is better than the worst model's loss, return True
            if loss < self.best_models[-1][0]:
                return True
            else:
                return False

    def save_model(self, model, epoch, loss):
        """Save the model if it's among the top-k based on the training loss."""
        # First, check if the model should be saved
        if self.check(loss):
            # If we have fewer than k models, simply append the model
            if len(self.best_models) < self.k:
                self.best_models.append((loss, epoch))
            else:
                # If the current loss is better than the worst model, replace it
                self.best_models.append((loss, epoch))

            # Sort by loss (ascending order) and remove worse models if necessary
            self.best_models.sort(key=lambda x: x[0])  # Sort by loss (ascending)

            # Save the model
            self._save_model(model, epoch, loss)

            # Remove worse models if there are more than k models
            self.remove_worse_models()

    def _save_model(self, model, epoch, loss):
        """Helper function to save the model to disk."""
        filename = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
        # Save the model state_dict
        torch.save(model.state_dict(), filename)
        print(f"Saved model to {filename}")

    def remove_worse_models(self):
        """Remove the worse models if there are more than k models."""
        # Ensure the list is sorted by the loss (ascending order)
        self.best_models.sort(key=lambda x: x[0])  # Sort by loss (ascending)

        # Remove models beyond the top-k
        while len(self.best_models) > self.k:
            loss, epoch = self.best_models.pop()
            filename = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed worse model {filename}")


def load_model(model, model_path, device):
    """Load a model's weights from a saved file with device compatibility."""
    # Map to the desired device (CPU or GPU)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


def data_loader(
    root_dir,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
):
    # Define the image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=True),  # Resize images to 128x128
            transforms.CenterCrop(64),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]
    )

    causal_celeba_dataset = CausalCelebA(
        root=root_dir,
        graph_type=graph_type,
        transform=image_transform,
        fast_dev_run=False,  # Set to True for debugging
    )

    # Calculate the number of samples for training and validation
    # total_len = len(causal_celeba_dataset)
    # val_len = int(total_len * val_split)
    # train_len = total_len - val_len

    # # Split the dataset into train and validation sets
    # train_dataset, val_dataset = random_split(causal_celeba_dataset, [train_len, val_len])

    distr_labels = [x[1][-1] for x in causal_celeba_dataset]
    unique_distrs = len(np.unique(distr_labels))
    if batch_size < unique_distrs:
        raise ValueError(
            f"Batch size must be at least {unique_distrs} for stratified sampling."
        )
    train_sampler = StratifiedSampler(distr_labels, batch_size)

    # Define the DataLoader
    train_loader = DataLoader(
        dataset=causal_celeba_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        # shuffle=True,  # Shuffle data during training
        num_workers=num_workers,
        pin_memory=True,  # Enable if using a GPU
    )

    return train_loader


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
    # root = Path("/home/adam2392/projects/data/")

    checkpoint_dir = root / "CausalCelebA" / "vae_reduction"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = 10_000
    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"
    debug = False
    num_workers = 6
    batch_size = 256
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

    model = VAE()
    model = model.to(device)
    image_dim = 3 * 64 * 64

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

    train_loader = data_loader(
        root_dir=root,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
    )

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
            f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}, LR: {lr:.6f}"
        )

        # Validation phase
        model.eval()

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
            if top_k_saver.check(model, train_loss, epoch):
                top_k_saver.save_model(model, epoch, train_loss)

    # Save top 5 models
    # for i, (val_loss, epoch, state_dict) in enumerate(top_models):
    #     torch.save(
    #         state_dict,
    #         f"{model_dir}/top_model_{i+1}_epoch_{epoch}_val_loss_{val_loss:.4f}.pth",
    #     )

    # Save final model
    # torch.save(model.state_dict(), checkpoint_dir / f"final_vaereduction_model.pt")
    # print(f"Training complete. Models saved in {checkpoint_dir}.")

    # # Usage example:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vae_model = VAE().to(device)
    # model_path = checkpoint_dir / f"final_vaereduction_model.pt"
    # vae_model = load_model(vae_model, model_path, device)
