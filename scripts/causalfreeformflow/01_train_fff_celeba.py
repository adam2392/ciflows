import os
from pathlib import Path

from torch import nn
import lightning as pl
import normflows as nf
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ciflows.datasets.causalceleba import CausalCelebA
from ciflows.datasets.multidistr import StratifiedSampler
from ciflows.distributions.pgm import LinearGaussianDag
from ciflows.eval import load_model
from ciflows.training import TopKModelSaver
from ciflows.loss import volume_change_surrogate
from ciflows.resnet_celeba import ResNetCelebADecoder, ResNetCelebAEncoder


class Freeformflow(nn.Module):
    def __init__(
        self,
        encoder: ResNetCelebAEncoder,
        decoder: ResNetCelebADecoder,
        latent: LinearGaussianDag,
    ) -> None:
        super(Freeformflow, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent
        self.latent_dim = encoder.latent_dim

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, distr_idx=None):
        z = self.latent.sample(num_samples, distr_idx=distr_idx)
        return self.decoder(z), z


def compute_loss(model: Freeformflow, x, distr_idx, beta):
    # calculate volume change surrogate loss
    surrogate_loss, v_hat, x_hat = volume_change_surrogate(
        images, model.encoder, model.decoder, hutchinson_samples=hutchinson_samples
    )

    # compute reconstruction loss
    loss_reconstruction = torch.nn.functional.mse_loss(x_hat, x)

    # get negative log likelihoood over the distributions
    embed_dim = model.decoder.latent_dim
    v_hat = v_hat.view(-1, embed_dim)
    loss_nll = (
        -model.latent.log_prob(v_hat, distr_idx=distr_idx).mean() - surrogate_loss
    )

    loss = beta * loss_reconstruction + loss_nll
    return loss, loss_reconstruction, loss_nll, surrogate_loss


def data_loader(
    root_dir,
    graph_type="chain",
    num_workers=4,
    batch_size=32,
    img_size=64,
):
    # Define the image transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),  # Resize images to 128x128
            transforms.CenterCrop(img_size),  # Ensure square crop
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    causal_celeba_dataset = CausalCelebA(
        root=root_dir,
        graph_type=graph_type,
        img_size=img_size,
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
        # pin_memory=True,  # Enable if using a GPU
    )

    return train_loader


def make_fff_model(debug=False):
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

    latent_dim = 48

    confounded_list = []
    # independent noise with causal prior
    latent = LinearGaussianDag(
        node_dimensions=node_dimensions,
        edge_list=edge_list,
        noise_means=noise_means,
        noise_variances=noise_variances,
        confounded_list=confounded_list,
        intervened_node_means=intervened_node_means,
        intervened_node_vars=intervened_node_vars,
    )

    # define the encoder and decoder
    encoder = ResNetCelebAEncoder(latent_dim=latent_dim)
    decoder = ResNetCelebADecoder(latent_dim=latent_dim)
    model = Freeformflow(encoder, decoder, latent=latent)
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
    img_size = 128
    max_epochs = 2000
    lr = 3e-4
    lr_min = 1e-8
    lr_scheduler = "cosine"
    max_norm = 1.0  # Threshold for gradient norm clipping
    debug = False
    num_workers = 2
    graph_type = "chain"

    if debug:
        accelerator = "cpu"
        device = "cpu"
        max_epochs = 5
        batch_size = 8
        check_samples_every_n_epoch = 1
        num_workers = 2

        fast_dev = True

    hutchinson_samples = 2
    beta = torch.tensor(10.0).to(device)

    torch.set_float32_matmul_precision("high")

    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
    else:
        root = Path("/home/adam2392/projects/data/")

    # v1: K=32
    # v2: K=8
    # v3: K=8, batch higher
    model_fname = "celeba_fff_batch256_latentdim48_v1.pt"

    checkpoint_dir = root / "CausalCelebA" / "fff" / model_fname.split(".")[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = make_fff_model(debug=debug)
    model = model.to(device)
    image_dim = 3 * img_size * img_size

    # compile the model
    # model = torch.compile(model)

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
        img_size=img_size,
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

        train_reconstruction_loss = 0.0
        train_nll_loss = 0.0
        train_surrogate_loss = 0.0
        for batch_idx, (images, distr_idx, targets, meta_labels) in tqdm(
            enumerate(train_loader), desc="step", position=1, leave=False
        ):
            # torch.cuda.empty_cache()
            images = images.to(device)
            optimizer.zero_grad()

            # compute the loss
            loss, loss_reconstruction, loss_nll, surrogate_loss = compute_loss(
                model, images, distr_idx, beta
            )

            loss = loss.mean()
            loss_reconstruction = loss_reconstruction.mean()
            loss_nll = loss_nll.mean()
            surrogate_loss = surrogate_loss.mean()

            # backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            # accumulate loss terms
            train_loss += loss.item()
            train_reconstruction_loss += loss_reconstruction.item()
            train_nll_loss += loss_nll.item()
            train_surrogate_loss += surrogate_loss.item()

        # Step the scheduler at the end of the epoch
        scheduler.step()

        train_loss /= len(train_loader)
        train_reconstruction_loss /= len(train_loader)
        train_nll_loss /= len(train_loader)
        train_surrogate_loss /= len(train_loader)
        lr = scheduler.get_last_lr()[0]
        print(
            f"====> Epoch: {epoch} Average loss: {train_loss:.4f}, LR: {lr:.6f}"
            f"Reconstruction Loss: {train_reconstruction_loss:.4f}, NLL Loss: {train_nll_loss:.4f}, Surrogate Loss: {train_surrogate_loss:.4f}"
        )

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
                # reconstruct images
                reconstructed_images, _ = model.sample(8, distr_idx=distr_idx)[0]

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
