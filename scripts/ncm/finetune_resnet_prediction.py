from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision import models

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Import user-defined dataset and model
from ciflows.datasets.causalmnistv2 import CausalMNIST_v2    # adjust import path as needed


class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features

        # digit prediction (classification)
        self.head_digit = nn.Linear(in_features, 10)

        # digit color regression (e.g., RGB float triplet)
        self.head_digit_color = nn.Linear(in_features, 3)

        # bar color regression (e.g., RGB float triplet)
        self.head_bar_color = nn.Linear(in_features, 3)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        return {
            "digit": self.head_digit(features),
            "digit_color": self.head_digit_color(features),
            "bar_color": self.head_bar_color(features),
        }


# ----- Training & Evaluation Functions -----
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, meta in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        targets = {
            'digit':       meta['digit'].to(device),
            'digit_color': meta['color_digit'].to(device),
            'bar_color':   meta['color_bar'].to(device)
        }
        outputs = model(imgs)
        loss = sum(criterion(outputs[k], targets[k]) for k in outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = {'digit': 0, 'digit_color': 0, 'bar_color': 0}
    total = 0
    with torch.no_grad():
        for imgs, meta in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device)
            targets = {
                'digit':       meta['digit'].to(device),
                'digit_color': meta['color_digit'].to(device),
                'bar_color':   meta['color_bar'].to(device)
            }
            outputs = model(imgs)
            total += imgs.size(0)
            for k in outputs:
                preds = outputs[k].argmax(dim=1)
                correct[k] += (preds == targets[k]).sum().item()
    return {k: correct[k] / total for k in correct}

if __name__ == "__main__":
    # ----- Configuration -----
    root_dir = "/path/to/data"                # root data directory
    distr_labels = ['observational', 'int_colorbar_0', 'int_colorbar_1', 'int_colorbar_2']                 # list of distribution labels
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3

    # Load device settings
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # accelerator = sys_cfg.get("accelerator", "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ----- Transforms -----
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ----- Datasets & Dataloaders -----
    train_ds = CausalMNIST_v2(root=root_dir, distr_labels=distr_labels, transform=transform)
    val_ds   = CausalMNIST_v2(root=root_dir, distr_labels=distr_labels, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # ----- Model, Loss, Optimizer -----
    model = MultiTaskCNN(num_digit_colors=5, num_bar_colors=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ----- Main Loop -----
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        logging.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Acc | digit: {val_acc['digit']:.4f}, "
            f"color_digit: {val_acc['digit_color']:.4f}, "
            f"color_bar: {val_acc['bar_color']:.4f}"
        )

    # Optionally: save model checkpoint
    checkpoint_path = Path("./checkpoints")
    checkpoint_path.mkdir(exist_ok=True)
    model_file = checkpoint_path / f"multitask_cnn_epoch{num_epochs}.pth"
    torch.save(model.state_dict(), model_file)
    logging.info(f"Model saved to {model_file}")
