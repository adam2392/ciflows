import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ciflows.vae import Encoder, Decoder


# Define the loss function (reconstruction + KL divergence)
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


if __name__ == "__main__":
    # Set up training configurations
    batch_size = 128
    learning_rate = 1e-3
    epochs = 1000  # Adjust as needed
    save_interval = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    encoder = Encoder(in_channels=1, img_size=28, embed_dim=32).to(
        device
    )  # Adjust embed_dim as needed
    decoder = Decoder(embed_dim=32, img_size=28, out_channels=1).to(device)

    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
    )

    # Training loop
    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Forward pass through the encoder
            mu, logvar = encoder(data)

            # Reparametrize: sample from the latent space
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Forward pass through the decoder
            recon_data = decoder(z)

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
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                },
                f"vae_checkpoint_epoch_{epoch+1}.pt",
            )
            print(f"Checkpoint saved at epoch {epoch+1}.")

            # Validation
            encoder.eval()
            decoder.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    mu, logvar = encoder(data)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    recon_data = decoder(z)

                    val_loss += loss_function(recon_data, data, mu, logvar).item()

            avg_val_loss = val_loss / len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

            encoder.train()
            decoder.train()

    print("Training complete.")
