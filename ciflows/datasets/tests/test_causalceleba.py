import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ciflows.datasets.causalceleba import CausalCelebA, CausalCelebAEmbedding


# Prepare a temporary directory structure with dummy data
def setup_dummy_data(root_dir):
    graph_type = "graph_type_dummy"
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
                "haircolor": np.random.randint(0, 5, size=10),
                "Intervention": np.repeat("haircolor", 10),
            }
        )
        causal_attrs.to_csv(distr_path / "causal_attrs.csv", index=False)

        # Create dummy meta_attrs.csv
        causal_attrs.to_csv(distr_path / "meta_attrs.csv", index=False)

        # Create dummy image files
        for i in range(10):
            img = PIL.Image.new("RGB", (64, 64), color=(i * 20 % 255, i * 40 % 255, i * 60 % 255))
            img.save(distr_path / f"sample_{i}.jpg")


def setup_dummy_data_causal_celeba_embedding(root_dir):
    """
    Creates dummy data for CausalCelebAEmbedding to test the dataset loader.

    Parameters
    ----------
    root_dir : str
        Path to the root directory where dummy data will be stored.
    """
    graph_type = "dummy_graph_type"
    distr_types = ["obs", "int_hair_0", "int_hair_1"]
    root_path = Path(root_dir) / "CausalCelebAEmbedding" / graph_type

    # Create dummy directories and files
    for distr_type in distr_types:
        distr_path = root_path / distr_type
        distr_path.mkdir(parents=True, exist_ok=True)

        # Dummy causal attributes CSV
        causal_attrs = pd.DataFrame(
            {
                "sample_index": np.random.randint(0, 2, size=10),
                "gender": np.random.randint(0, 2, size=10),
                "age": np.random.randint(18, 60, size=10),
                "haircolor": np.random.randint(0, 5, size=10),
                "Intervention": np.repeat("haircolor", 10),
            }
        )
        causal_attrs.to_csv(distr_path / "causal_attrs.csv", index=False)

        # Dummy meta attributes CSV
        meta_attrs = pd.DataFrame(
            {
                "meta_1": [1, 2],
                "meta_2": [3, 4],
            }
        )
        meta_attrs.to_csv(distr_path / "meta_attrs.csv", index=False)

        # Dummy image encodings
        encodings = torch.rand(10, 128)  # Example: 10 samples, 128-dimensional embeddings
        torch.save(encodings, distr_path / f"{distr_type}_encodings.pt")

    # Create intervention targets
    intervention_targets = torch.randint(0, 2, (10, 3))  # 6 samples, 3 latent variables
    torch.save(intervention_targets, root_path / "intervention_targets.pt")


# Test the dataloader
def test_dataloader():
    root_dir = tempfile.mkdtemp()  # Create a temporary directory
    try:
        setup_dummy_data(root_dir)  # Set up dummy data

        # Instantiate the dataloader
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CausalCelebA(
            root=root_dir,
            graph_type="graph_type_dummy",
            transform=transform,
            fast_dev_run=True,
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Print some dataset stats
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 30
        for i, (img, meta_label, target) in enumerate(dataloader):
            print(f"Batch {i}")
            print(f"Images shape: {img.shape}")
            print(f"Meta labels: {meta_label}")
            print(f"Targets: {target}")
            assert img.shape == (2, 3, 64, 64)
            assert np.asarray(meta_label).T.shape == (2, len(dataset.meta_label_strs))
            assert target.shape == (2, 3)
            if i == 2:  # Test a few batches only
                break

    finally:
        shutil.rmtree(root_dir)  # Clean up temporary directory


def test_causal_celeba_embedding():
    # Create a temporary directory for dummy data
    root_dir = tempfile.mkdtemp()

    try:
        # Setup dummy data
        setup_dummy_data_causal_celeba_embedding(root_dir)

        # Instantiate the dataset
        dataset = CausalCelebAEmbedding(
            root=root_dir,
            graph_type="dummy_graph_type",
            fast_dev_run=True,
        )

        # Test dataset length
        assert len(dataset) == 30, f"Expected dataset length of 30, got {len(dataset)}."

        # Test a single sample
        img, meta_label, target = dataset[0]
        assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor."
        assert isinstance(meta_label, list), "Meta-label should be a list."
        assert isinstance(target, torch.Tensor), "Target should be a torch.Tensor."
        assert img.shape[0] == 128, f"Image should have 128 dims {img.shape}."
        assert len(meta_label) == 6, "Meta-label should have 3 elements."

        print("Single sample test passed!")

        # Test DataLoader integration
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        for batch in dataloader:
            imgs, meta_labels, targets = batch
            assert imgs.shape[0] == 2, "Batch size should be 2."
            assert len(meta_labels) == 6, "Meta-labels should be 6"
            assert targets.shape[0] == 2, "Targets should match batch size."
            break  # Only test the first batch

        print("DataLoader test passed!")

    finally:
        # Clean up temporary directory
        shutil.rmtree(root_dir)
