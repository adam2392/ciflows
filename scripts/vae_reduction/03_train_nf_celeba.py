import os
from pathlib import Path

import lightning as pl
import normflows as nf
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ciflows.datasets.causalceleba import CausalCelebAEmbedding
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.eval import load_model
from ciflows.flows.model import CausalNormalizingFlow
from ciflows.reduction.vae import VAE
from ciflows.reduction.resnetvae import DeepResNetVAE


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


def data_loader(
    root_dir,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
    image_size=64,
):
    # Define the image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), antialias=True
            ),  # Resize images to 128x128
            transforms.CenterCrop(image_size),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    causal_celeba_dataset = CausalCelebAEmbedding(
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

    distr_labels = [x[1] for x in causal_celeba_dataset]
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


def make_nf_model(debug=False):
    """Make normalizing flow model."""
    # Define list of flows
    if debug:
        K = 2
        net_hidden_layers = 2
        net_hidden_dim = 64
    else:
        K = 8
        net_hidden_layers = 3
        net_hidden_dim = 64

    latent_dim = 48

    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_dim, net_hidden_layers, net_hidden_dim
            )
        ]

    node_dimensions = {
        0: 16,
        1: 16,
        2: 16,
    }
    edge_list = [(1, 2)]
    noise_means = {
        0: torch.zeros(16),
        1: torch.zeros(16),
        2: torch.zeros(16),
    }
    noise_variances = {
        0: torch.ones(16),
        1: torch.ones(16),
        2: torch.ones(16),
    }
    intervened_node_means = [{2: torch.ones(16) + 2}, {2: torch.ones(16) + 4}]
    intervened_node_vars = [{2: torch.ones(16)}, {2: torch.ones(16) + 2}]

    confounded_list = []
    # independent noise with causal prior
    q0 = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
    )

    # Construct flow model with the multiscale architecture
    model = CausalNormalizingFlow(q0, flows)
    return model


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

    batch_size = 256

    max_epochs = 2000
    lr = 3e-4
    lr_min = 1e-6
    lr_scheduler = "cosine"
    max_norm = 1.0  # Threshold for gradient norm clipping
    debug = False
    num_workers = 6
    graph_type = "chain"
    image_size = 128
    latent_dim = 48
    num_blocks_per_stage = 3

    torch.set_float32_matmul_precision("high")

    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
    else:
        root = Path("/home/adam2392/projects/data/")

    # v1: K=32
    # v2: K=8
    # v3: K=8, batch higher
    model_fname = "celeba_nfon_resnetvaereduction_batch256_latentdim48_v1.pt"

    # checkpoint_dir = root / "CausalCelebA" / "vae_reduction" / "latentdim24"
    checkpoint_dir = (
        root / "CausalCelebA" / "nf_on_vae_reduction" / model_fname.split(".")[0]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # vae_dir = root / "CausalCelebA" / "vae_reduction" / "latentdim48"
    # vae_model_fname = "model_epoch_100.pt"
    vae_model_fname = "celeba_vaeresnetreduction_batch512_latentdim48_img128_v1.pt"
    vae_dir = root / "CausalCelebA" / "vae_reduction" / vae_model_fname.split(".")[0]
    # vae_model = VAE().to(device)
    vae_model = DeepResNetVAE(latent_dim, num_blocks_per_stage=num_blocks_per_stage)
    model_path = vae_dir / vae_model_fname
    vae_model = load_model(vae_model, model_path, device)
    vae_model = vae_model.to(device)

    if debug:
        accelerator = "cpu"
        # device = 'cpu'
        max_epochs = 5
        batch_size = 8
        check_samples_every_n_epoch = 1
        num_workers = 2

        fast_dev = True

    model = make_nf_model(debug=debug)
    model = model.to(device)
    image_dim = 3 * image_size * image_size

    # compile the model
    model = torch.compile(model)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Cosine Annealing Scheduler (adjust the T_max for the number of epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr_min
    )  # T_max = total epochs

    top_k_saver = TopKModelSaver(
        checkpoint_dir, k=5
    )  # Initialize the top-k model saver

    train_loader = data_loader(
        root_dir=root,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
        image_size=image_size,
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
        for batch_idx, (images, distr_idx, targets, meta_labels) in tqdm(
            enumerate(train_loader), desc="step", position=1, leave=False
        ):
            # torch.cuda.empty_cache()
            images = images.to(device)
            optimizer.zero_grad()

            # extract data from tensor to Parameterdict
            loss = model.forward_kld(
                images, intervention_targets=targets, distr_idx=distr_idx
            )

            # backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_loss += loss.item()

            if debug:
                break

        # Step the scheduler at the end of the epoch
        scheduler.step()

        train_loss /= len(train_loader)
        lr = scheduler.get_last_lr()[0]
        print(f"====> Epoch: {epoch} Average loss: {train_loss:.4f}, LR: {lr:.6f}")

        # Validation phase
        model.eval()

        # Log training and validation loss
        if debug or epoch % 10 == 0:
            print()
            print(
                f"Saving images - Epoch [{epoch}/{max_epochs}], Val Loss: {train_loss:.4f}"
            )

            # sample images from normalizing flow
            for distr_idx in train_loader.dataset.distr_idx_list:
                sample_embeddings, _ = model.sample(8, distr_idx=distr_idx)

                # reconstruct images
                reconstructed_images = vae_model.decode(sample_embeddings).reshape(
                    -1, 3, image_size, image_size
                )

                save_image(
                    reconstructed_images.cpu(),
                    checkpoint_dir / f"epoch_{epoch}_distr-{distr_idx}_samples.png",
                    nrow=4,
                    normalize=True,
                )

        # Track top 5 models based on validation loss
        if epoch % 5 == 0:
            # Optionally, remove worse models if there are more than k saved models
            top_k_saver.save_model(model, epoch, train_loss)

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / model_fname)
    print(f"Training complete. Models saved in {checkpoint_dir}.")

    # Usage example:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf_model = model.to(device)
    model_path = checkpoint_dir / model_fname
    nf_model = load_model(nf_model, model_path, device)
