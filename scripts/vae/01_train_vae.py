from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ciflows.vae import Conv_VAE


# Define the loss function (reconstruction + KL divergence)
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


class VAE(nn.Module):
    def __init__(self, latent_dim=20, dropout=0.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Input: 1x28x28
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Input: 32x14x14
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),  # Outputs mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Input: 64x7x7
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),  # Input: 32x14x14
            nn.Sigmoid(),  # Output: 1x28x28
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = h[:, : self.latent_dim], h[:, self.latent_dim :]
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var


if __name__ == "__main__":
    # Set up training configurations
    batch_size = 128
    learning_rate = 1e-3
    epochs = 1000  # Adjust as needed
    save_interval = 25

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    # MNIST dataset loader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize your Encoder and Decoder
    channels = 1  # For grayscale images (like MNIST); set to 3 for RGB (like CelebA)
    height = 28  # Height of the input image (28 for MNIST)
    width = 28  # Width of the input image (28 for MNIST)
    model = Conv_VAE(channels=channels, height=height, width=width, hidden_size=16).to(
        device
    )
    # encoder = Encoder(
    #     in_channels=1,
    #     img_size=28,
    #     embed_dim=32,
    #     hidden_dim=1024,
    #     n_bottleneck_layers=5,
    #     n_compression_layers=2,
    # ).to(
    #     device
    # )  # Adjust embed_dim as needed
    # decoder = Decoder(
    #     embed_dim=32,
    #     img_size=28,
    #     hidden_dim=1024,
    #     out_channels=1,
    #     n_bottleneck_layers=5,
    #     n_upsample_layers=2,
    # ).to(device)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get the number of trainable parameters in the encoder and decoder
    num_model_params = count_trainable_parameters(model)
    # num_decoder_params = count_trainable_parameters(decoder)

    # Print the results
    print(f"Total trainable parameters in Encoder: {num_model_params}")

    # Optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        train_loss = 0

        for batch_idx, (data, _) in tqdm(
            enumerate(train_loader),
            desc="Batches",
            total=len(train_loader),
            position=1,
            leave=False,
        ):
            data = data.to(device)

            # Forward pass through the encoder
            recon_data, mu, logvar = model(data)

            # Compute the loss
            loss = loss_function(recon_data, data, mu, logvar)
            train_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # Save the model and validate every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            results_path = Path("./results/vae/")
            results_path.mkdir(exist_ok=True, parents=True)
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                },
                results_path / f"vae_checkpoint_epoch_{epoch+1}.pt",
            )
            print(f"Checkpoint saved at epoch {epoch+1}.")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    recon_data, mu, logvar = model(data)

                    val_loss += loss_function(recon_data, data, mu, logvar).item()

            avg_val_loss = val_loss / len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

            model.train()

    print("Training complete.")
