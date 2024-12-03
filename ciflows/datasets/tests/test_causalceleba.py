import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
from torch.utils.data import DataLoader
from torchvision import transforms

from ciflows.datasets.causalceleba import CausalCelebA


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
                "intervention": np.repeat("haircolor", 10),
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
