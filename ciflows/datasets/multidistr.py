import numpy as np
from torch.utils.data import Sampler


class StratifiedSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(labels)
        self.unique_labels = np.unique(labels)
        self.label_indices = {
            label: np.where(np.array(labels) == label)[0]
            for label in self.unique_labels
        }
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        num_per_class = self.batch_size // len(self.unique_labels)

        for _ in range(self.num_samples // self.batch_size):
            batch_indices = []
            for label in self.unique_labels:
                label_indices = np.random.choice(
                    self.label_indices[label], num_per_class, replace=False
                )
                batch_indices.extend(label_indices)

            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)

        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
