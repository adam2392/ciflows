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

    max_epochs = 1000
    lr = 3e-4
    lr_min = 1e-8
    max_norm = 1.0  # Threshold for gradient norm clipping
    debug = False
    num_workers = 10
    graph_type = "chain"

    torch.set_float32_matmul_precision("high")

    if debug:
        root = Path("/Users/adam2392/pytorch_data/")
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
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    top_k_saver = TopKModelSaver(
        checkpoint_dir, k=5
    )  # Initialize the top-k model saver

    # Load CelebA dataset
    train_dataset = CelebA(root=root, split="train", transform=transform, download=True)
    val_dataset = CelebA(root=root, split="valid", transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    # Loss functions and metrics
    loss_gender = nn.CrossEntropyLoss()
    loss_hair = nn.CrossEntropyLoss()
    loss_age = nn.CrossEntropyLoss()

    acc_gender = Accuracy(task="binary").to(device)
    acc_hair = Accuracy(task="multiclass", num_classes=4).to(device)
    acc_age = Accuracy(task="binary").to(device)

    # Initialize the model, optimizer, and scheduler
    model = MultiTaskResNet().to(device)
    # model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr_min
    )  # Adjust T_max as needed

    # TensorBoard logging setup
    writer = SummaryWriter(log_dir)

    attr_names = train_dataset.attr_names
    gender_idx = train_dataset.attr_names.index("Male")
    age_idx = train_dataset.attr_names.index("Young")
    blackhair_idx = attr_names.index("Black_Hair")
    blondhair_idx = attr_names.index("Blond_Hair")
    brownhair_idx = attr_names.index("Brown_Hair")
    grayhair_idx = attr_names.index("Gray_Hair")
    hair_cols = np.array([blackhair_idx, blondhair_idx, brownhair_idx, grayhair_idx])

    # create a dictionary of the attribute names
    model.age_target_map = {
        "Old": 0,
        "Young": 1,
    }
    model.gender_target_map = {
        "Female": 0,
        "Male": 1,
    }
    model.hair_target_map = {
        "Black_Hair": 0,
        "Blond_Hair": 1,
        "Brown_Hair": 2,
        "Gray_Hair": 3,
    }

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            gender, hair, age = (
                labels[:, gender_idx],
                labels[:, hair_cols],
                labels[:, age_idx],
            )

            # 0: black
            # 1: blond
            # 2: brown
            # 3: gray
            hair = torch.argmax(hair, axis=1)

            optimizer.zero_grad()

            # Forward pass
            (gender_out, gender_prob), (hair_out, hair_prob), (age_out, age_prob) = (
                model(images)
            )

            # Calculate loss
            loss_g = loss_gender(gender_out, gender)
            loss_h = loss_hair(hair_out, hair)
            loss_a = loss_age(age_out, age)

            total_loss = loss_g + loss_h + loss_a
            total_loss.backward()

            optimizer.step()

            running_loss += total_loss.item()

            # Update metrics
            # print(gender_prob.shape, gender.shape)
            # print(gender_prob, torch.argmax(gender_prob, dim=1))
            acc_gender.update(torch.argmax(gender_prob, dim=1), gender)
            acc_hair.update(torch.argmax(hair_prob, dim=1), hair)
            acc_age.update(torch.argmax(age_prob, dim=1), age)

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

        if debug or epoch % check_samples_every_n_epoch == 0:
            # Reset metrics for the validation
            acc_gender.reset()
            acc_hair.reset()
            acc_age.reset()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    gender, hair, age = labels["gender"], labels["hair"], labels["age"]

                    # Forward pass
                    (
                        (gender_out, gender_prob),
                        (hair_out, hair_prob),
                        (age_out, age_prob),
                    ) = model(images)

                    # Calculate loss
                    loss_g = loss_gender(gender_out, gender)
                    loss_h = loss_hair(hair_out, hair)
                    loss_a = loss_age(age_out, age)

                    val_loss += (loss_g + loss_h + loss_a).item()

                    # Update metrics
                    acc_gender.update(torch.argmax(gender_prob, dim=1), gender)
                    acc_hair.update(torch.argmax(hair_prob, dim=1), hair)
                    acc_age.update(torch.argmax(age_prob, dim=1), age)

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
