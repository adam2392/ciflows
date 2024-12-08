from enum import Enum
from pathlib import Path
from typing import Optional

import lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .causalceleba import CausalCelebA
from .causalmnist import CausalDigitBarMNIST
from .multidistr import StratifiedSampler


class DatasetName(Enum):
    CAUSAL_DIGIT_BAR_MNIST = "causal_digit_bar_mnist"
    CAUSAL_CELEBA = "causal_celeba"  # Add more datasets as needed


class MultiDistrDataModule(pl.LightningDataModule):
    """
    Data module for multi-distributional data.

    Attributes
    ----------
    medgp: MultiEnvDGP
        Multi-environment data generating process.
    num_samples_per_env: int
        Number of samples per environment.
    batch_size: int
        Batch size.
    num_workers: int
        Number of workers for the data loaders.
    intervention_targets_per_distr: Tensor, shape (num_envs, num_causal_variables)
        Intervention targets per environment, with 1 indicating that the variable is intervened on.
    log_dir: Optional[Path]
        Directory to save summary statistics and plots to. Default: None.
    intervention_target_misspec: bool
        Whether to misspecify the intervention targets. If true, the intervention targets are permuted.
        I.e. the model received the wrong intervention targets. Default: False.
    intervention_target_perm: Optional[list[int]]
        Permutation of the intervention targets. If None, a random permutation is used. Only used if
        intervention_target_misspec is True. Default: None.
    flatten: bool
        Whether to flatten the data. Default: False.

    Methods
    -------
    setup(stage=None) -> None
        Setup the data module. This is where the data is sampled.
    train_dataloader() -> DataLoader
        Return the training data loader.
    val_dataloader() -> DataLoader
        Return the validation data loader.
    test_dataloader() -> DataLoader
        Return the test data loader.
    """

    def __init__(
        self,
        root,
        graph_type,
        batch_size: int,
        img_size: int = 64,
        stratify_distrs: bool = True,
        num_workers: int = -1,
        train_size: float = 0.8,
        val_size: float = 0.1,
        log_dir: Optional[Path] = None,
        transform=None,
        dataset_name: DatasetName = DatasetName.CAUSAL_DIGIT_BAR_MNIST,
        fast_dev_run: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir) if log_dir is not None else log_dir

        self.img_size = img_size
        self.stratify_distrs = stratify_distrs
        self.train_size = train_size
        self.val_size = val_size

        self.root = root
        self.graph_type = graph_type
        self.fast_dev_run = fast_dev_run

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((32, 32)),
                    # discretize,
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name == DatasetName.CAUSAL_DIGIT_BAR_MNIST:
            self.dataset = CausalDigitBarMNIST(
                root=self.root,
                graph_type=self.graph_type,
                transform=self.transform,
                fast_dev_run=self.fast_dev_run,
            )
        elif self.dataset_name == DatasetName.CAUSAL_CELEBA:
            self.dataset = CausalCelebA(
                root=self.root,
                graph_type=self.graph_type,
                transform=self.transform,
                img_size=self.img_size,
                fast_dev_run=self.fast_dev_run,
            )

        train_size = int(self.train_size * len(self.dataset))
        val_size = int(self.val_size * (len(self.dataset) - train_size))
        test_size = len(self.dataset) - train_size - val_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_split(self.dataset, [train_size, val_size, test_size])

        if self.stratify_distrs:
            distr_labels = [x[1] for x in self.train_dataset]
            unique_distrs = len(np.unique(distr_labels))
            if self.batch_size < unique_distrs:
                raise ValueError(
                    f"Batch size must be at least {unique_distrs} for stratified sampling."
                )
            self.train_sampler = StratifiedSampler(distr_labels, self.batch_size)

            distr_labels = [x[1] for x in self.val_dataset]
            unique_distrs = len(np.unique(distr_labels))
            if self.batch_size < unique_distrs:
                raise ValueError(
                    f"Batch size must be at least {unique_distrs} for stratified sampling."
                )
            self.val_sampler = StratifiedSampler(distr_labels, self.batch_size)
        else:
            self.train_sampler = None
            self.val_sampler = None

    @property
    def meta_label_strs(self):
        return self.dataset.meta_label_strs

    @property
    def latent_dim(self):
        return self.dataset.latent_dim

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return test_loader
