from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy
from torchvision import transforms
from torchvision.datasets import CelebA

from ciflows.datasets.causalceleba_scm.pretrained import MultiTaskResNet
from ciflows.eval import load_model
from ciflows.training import TopKModelSaver

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

max_epochs = 1000
lr = 3e-4
lr_min = 1e-8
max_norm = 1.0  # Threshold for gradient norm clipping
debug = True
num_workers = 10
graph_type = "chain"

torch.set_float32_matmul_precision("high")

if debug:
    root = Path("/Users/adam2392/pytorch_data/celeba")
else:
    root = Path("/home/adam2392/projects/data/")

# v1: K=32
# v2: K=8
# v3: K=8, batch higher
model_fname = "celeba_predictor_batch256_v1.pt"

# checkpoint_dir = root / "CausalCelebA" / "vae_reduction" / "latentdim24"
checkpoint_dir = root / "CausalCelebA" / "pretrained" / model_fname.split(".")[0]
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir = checkpoint_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

if debug:
    accelerator = "cpu"
    # device = 'cpu'
    max_epochs = 5
    batch_size = 256
    check_samples_every_n_epoch = 1
    num_workers = 2

    fast_dev = True

# Data preparation and augmentation
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

top_k_saver = TopKModelSaver(checkpoint_dir, k=5)  # Initialize the top-k model saver


# Load CelebA dataset
train_dataset = CelebA(root=root, split="train", transform=transform, download=True)
val_dataset = CelebA(root=root, split="valid", transform=transform, download=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True,
)

# Loss functions and metrics
loss_gender = nn.CrossEntropyLoss()
loss_hair = nn.CrossEntropyLoss()
loss_age = nn.CrossEntropyLoss()

acc_gender = Accuracy()
acc_hair = Accuracy()
acc_age = Accuracy()

# Initialize the model, optimizer, and scheduler
model = MultiTaskResNet().to(device)
model = torch.compile(model)

optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epochs, eta_min=lr_min
)  # Adjust T_max as needed

# TensorBoard logging setup
writer = SummaryWriter(log_dir)

# Training loop
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        gender, hair, age = labels["gender"], labels["hair"], labels["age"]

        optimizer.zero_grad()

        # Forward pass
        gender_out, hair_out, age_out = model(images)

        # Calculate loss
        loss_g = loss_gender(gender_out, gender)
        loss_h = loss_hair(hair_out, hair)
        loss_a = loss_age(age_out, age)

        total_loss = loss_g + loss_h + loss_a
        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

        # Update metrics
        acc_gender.update(gender_out, gender)
        acc_hair.update(hair_out, hair)
        acc_age.update(age_out, age)

    scheduler.step()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_acc_gender = acc_gender.compute()
    avg_train_acc_hair = acc_hair.compute()
    avg_train_acc_age = acc_age.compute()

    # Log training results to TensorBoard
    writer.add_scalar("train_loss", avg_train_loss, epoch)
    writer.add_scalar("train_acc_gender", avg_train_acc_gender, epoch)
    writer.add_scalar("train_acc_hair", avg_train_acc_hair, epoch)
    writer.add_scalar("train_acc_age", avg_train_acc_age, epoch)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

    # Log gradients (optional)
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            gender, hair, age = labels["gender"], labels["hair"], labels["age"]

            gender_out, hair_out, age_out = model(images)

            # Calculate loss
            loss_g = loss_gender(gender_out, gender)
            loss_h = loss_hair(hair_out, hair)
            loss_a = loss_age(age_out, age)

            val_loss += (loss_g + loss_h + loss_a).item()

            # Update metrics
            acc_gender.update(gender_out, gender)
            acc_hair.update(hair_out, hair)
            acc_age.update(age_out, age)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc_gender = acc_gender.compute()
    avg_val_acc_hair = acc_hair.compute()
    avg_val_acc_age = acc_age.compute()

    # Log validation results to TensorBoard
    writer.add_scalar("val_loss", avg_val_loss, epoch)
    writer.add_scalar("val_acc_gender", avg_val_acc_gender, epoch)
    writer.add_scalar("val_acc_hair", avg_val_acc_hair, epoch)
    writer.add_scalar("val_acc_age", avg_val_acc_age, epoch)

    # Reset metrics for the next epoch
    acc_gender.reset()
    acc_hair.reset()
    acc_age.reset()

    print(
        f"====> Epoch {epoch+1}/{max_epochs}, "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Train Acc (Gender): {avg_train_acc_gender:.4f}, "
        f"Val Acc (Gender): {avg_val_acc_gender:.4f}"
    )

    # Track top 5 models based on validation loss
    if epoch % 5 == 0:
        # Optionally, remove worse models if there are more than k saved models
        top_k_saver.save_model(model, epoch, avg_train_loss)

# Close the TensorBoard writer after training is done
writer.close()


# Save final model
torch.save(model.state_dict(), checkpoint_dir / model_fname)
print(f"Training complete. Models saved in {checkpoint_dir}.")

# Usage example:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nf_model = model.to(device)
model_path = checkpoint_dir / "final_nf_model.pt"
nf_model = load_model(nf_model, model_path, device)
