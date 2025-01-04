import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler


class StratifiedSampler(Sampler):
    def __init__(self, labels, batch_size, seed=None):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(labels)
        self.unique_labels = np.unique(labels)
        self.label_indices = {
            label: np.where(np.array(labels) == label)[0] for label in self.unique_labels
        }
        self.rng = np.random.default_rng(seed)
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        num_per_class = self.batch_size // len(self.unique_labels)

        for _ in range(self.num_samples // self.batch_size):
            batch_indices = []
            for label in self.unique_labels:
                label_indices = self.rng.choice(
                    self.label_indices[label], num_per_class, replace=False
                )
                batch_indices.extend(label_indices)

            self.rng.shuffle(batch_indices)
            indices.extend(batch_indices)

        return indices

    def __iter__(self):
        # shuffle each time a new iterator is called
        self.indices = self._generate_indices()
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class SynchronizedDataset(Dataset):
    def __init__(self, image_dataset, embedding_dataset):
        """
        Dataset that synchronizes access to images and their corresponding embeddings.

        Args:
            image_dataset: Original dataset containing images
            embedding_dataset: Dataset containing pre-computed embeddings
        """
        self.image_dataset = image_dataset
        self.embedding_dataset = embedding_dataset
        assert len(image_dataset) == len(embedding_dataset), "Datasets must have same length"

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        """
        Returns both image and its corresponding embedding for the same index.
        """
        image_data = self.image_dataset[idx]
        embedding_data = self.embedding_dataset[idx]
        return {"image": image_data, "embedding": embedding_data, "index": idx}
