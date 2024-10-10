from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle=True,
        fast_dev_run=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.fast_dev_run = fast_dev_run

    def setup(self, stage: str):
        self.mnist_test = MNIST(
            self.data_dir, download=True, train=False, transform=self.transform
        )
        mnist_full = MNIST(
            self.data_dir, download=True, train=True, transform=self.transform
        )
        if self.fast_dev_run:
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [100, 60_000 - 100],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
