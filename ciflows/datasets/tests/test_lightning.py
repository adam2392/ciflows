import shutil
import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import PIL
import pytest
import torch
from torchvision import transforms

from ciflows.datasets.lightning import DatasetName, MultiDistrDataModule


def setup_dummy_data_causal_digitbar_mnist(root_dir, graph_type="graph_type_dummy"):
    """
    Creates dummy data for CausalDigitBarMNIST to test the dataset loader.

    Parameters
    ----------
    root_dir : str
        Path to the root directory where dummy data will be stored.
    """
    num_samples = 100  # Number of dummy samples
    num_features = 3 * 28 * 28  # Assuming flattened 28x28 images

    # Define paths
    dataset_path = Path(root_dir) / "CausalDigitBarMNIST" / graph_type
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Create dummy image data (num_samples x num_features)
    dummy_images = torch.rand(num_samples, num_features).reshape(-1, 3, 28, 28)
    torch.save(dummy_images, dataset_path / f"{graph_type}-imgs-train.pt")

    # Create dummy labels (num_samples x 4) -> digit, color_digit, color_bar, distr_idx
    dummy_labels = torch.randint(0, 10, (num_samples, 1))  # Digits 0-9
    dummy_color_digit = torch.randint(
        0, 3, (num_samples, 1)
    )  # 3 color labels for digit
    dummy_color_bar = torch.randint(0, 3, (num_samples, 1))  # 3 color labels for bar
    dummy_distr_idx = torch.randint(0, 4, (num_samples, 1))  # 4 distribution indices
    dummy_meta_labels = torch.hstack(
        [dummy_labels, dummy_color_digit, dummy_color_bar, dummy_distr_idx]
    )
    torch.save(dummy_meta_labels, dataset_path / f"{graph_type}-labels-train.pt")

    # Create dummy intervention targets (num_samples x latent_dim)
    dummy_intervention_targets = torch.randint(
        0, 2, (num_samples, 4)
    )  # 4 latent variables
    torch.save(
        dummy_intervention_targets, dataset_path / f"{graph_type}-targets-train.pt"
    )


# Prepare a temporary directory structure with dummy data
def setup_dummy_data_causalceleba(root_dir, graph_type="graph_type_dummy"):

    distr_types = ["obs", "int_hair_0", "int_hair_1"]

    for distr_type in distr_types:
        distr_path = Path(root_dir) / "CausalCelebA" / graph_type / distr_type
        distr_path.mkdir(parents=True, exist_ok=True)

        # Create dummy causal_attrs.csv
        causal_attrs = pd.DataFrame(
            {
                "sample_index": np.random.randint(0, 2, size=10),
                "gender": np.random.randint(0, 2, size=10),
                "age": np.random.randint(18, 60, size=10),
                "Haircolor": np.random.randint(0, 5, size=10),
                "Intervention": np.repeat("Haircolor", 10),
            }
        )
        causal_attrs.to_csv(distr_path / "causal_attrs.csv", index=False)

        # Create dummy meta_attrs.csv
        causal_attrs.to_csv(distr_path / "meta_attrs.csv", index=False)

        # Create dummy image files
        for i in range(10):
            img = PIL.Image.new(
                "RGB", (64, 64), color=(i * 20 % 255, i * 40 % 255, i * 60 % 255)
            )
            img.save(distr_path / f"sample_{i}.jpg")


@pytest.mark.parametrize(
    "dataset_name",
    [
        DatasetName.CAUSAL_CELEBA,
        # DatasetName.CAUSAL_DIGIT_BAR_MNIST
    ],
)
def test_multidistr_datamodule(dataset_name):
    # Create a temporary directory for dummy data
    root_dir = tempfile.mkdtemp()
    batch_size = 4
    graph_type = "graph_type_dummy"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            # discretize,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    try:
        # Setup dummy data
        if dataset_name == DatasetName.CAUSAL_DIGIT_BAR_MNIST:
            setup_dummy_data_causal_digitbar_mnist(root_dir, graph_type=graph_type)
        else:
            setup_dummy_data_causalceleba(root_dir, graph_type=graph_type)

        # Initialize the data module
        data_module = MultiDistrDataModule(
            root=root_dir,
            graph_type=graph_type,
            batch_size=batch_size,
            stratify_distrs=True,
            num_workers=0,  # Set to 0 for testing
            train_size=0.5,
            val_size=0.1,
            dataset_name=dataset_name,
            transform=transform,
            fast_dev_run=True,
        )

        # Setup the datasets
        data_module.setup()

        # Test train dataloader
        train_loader = data_module.train_dataloader()
        print("Train Dataloader:")
        for i, (img, meta_label, target) in enumerate(train_loader):
            print(f"Batch {i}")
            print(f"Images: {img.shape}")
            print(f"Meta labels: {meta_label}")
            print(f"Targets: {target}", target.shape)

            assert img.shape == (batch_size, 3, 64, 64)
            assert np.asarray(meta_label).T.shape == (
                batch_size,
                len(data_module.meta_label_strs),
            )

            assert_array_equal(target.shape, (batch_size, data_module.latent_dim))
            assert_array_equal(np.unique(target), np.array([0, 1]))
            if i == 2:  # Test a few batches
                break

    finally:
        # Clean up temporary directory
        shutil.rmtree(root_dir)
