from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
import PIL


class CausalMNIST_v2(Dataset):
    def __init__(
        self,
        root,
        distr_labels,
        # graph_type,
        transform=None,
        fast_dev_run=False,
    ):
        """A multidistributional dataset for colored MNIST images with a colored bar.

        This dataset contains multiple distributions of MNIST images, each with different
        interventions applied, possibly arising from different domains.
        """
        self.root = Path(root) / self.__class__.__name__
        self.transform = transform
        # self.graph_type = graph_type

        root = Path(root)

        # load data from disc
        distr_dirs = [self.root / f"{distr_labels[i]}" for i in range(len(distr_labels))]
        self.labels = pd.concat(
            [pd.read_csv(d / "causal_labels.csv") for d in distr_dirs], ignore_index=True
        )
        self.intervention_targets = torch.cat(
            [torch.load(d / "targets.pt") for d in distr_dirs], dim=0
        )

        # check number of images in each distribution
        # and preload the image files into a list
        self.file_list = []
        for d in distr_dirs:
            img_files = list(d.glob("images/*.png"))

            if not img_files:
                raise ValueError(f"No images found in {d}")
            self.file_list.extend(img_files)

        if len(self.file_list) != len(self.labels):
            raise ValueError(
                f"Number of images ({len(self.file_list)}) does not match number of labels ({len(self.labels)})"
            )

        # self.labels = torch.load(
        #     root / self.__class__.__name__ /  / f"{graph_type}-labels-train.pt",
        #     weights_only=False,
        # )
        # if isinstance(self.labels, list):
        #     self.labels = torch.vstack(self.labels)

        # self.intervention_targets = torch.load(
        #     root / self.__class__.__name__ / graph_type / f"{graph_type}-targets-train.pt",
        #     weights_only=False,
        # )
        # if isinstance(self.intervention_targets, list):
        #     self.intervention_targets = torch.vstack(self.intervention_targets)

        if fast_dev_run:
            subsample = 100
            # self.data = self.data[:subsample]
            self.labels = self.labels[:subsample]
            self.intervention_targets = self.intervention_targets[:subsample]
            self.file_list = self.file_list[:subsample]

    @property
    def intervention_targets_per_distr(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
        ]

    def get_distribution_labels(self):
        """Get the distribution labels for each image."""
        return self.labels['distr_idx'].tolist()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        """Get a sample from the image dataset.

        The target composes of the meta-labeling:
        - gender
        - age
        - haircolor

        Returns
        -------
        img : torch.Tensor of shape (C, H, W)
            Image tensor
        meta_label : list
            List of meta-labels
        target : torch.Tensor of shape (latent_dim,)
            Intervention target with 1's where the intervention is applied.
        """
        filename = self.file_list[index]
        img = PIL.Image.open(filename)

        metadata_for_img = self.labels.iloc[index]
        distr_idx = metadata_for_img["distr_idx"]

        # XXX: only can handle one type of intervention
        target = self.intervention_targets[index]

        # only extract the array
        # metadata_for_img = metadata_for_img.values.tolist()

        if self.transform is not None:
            img = self.transform(img)

        meta_dict = {
            'distribution_idx': distr_idx,
            'digit': metadata_for_img['digit'],
            'color_digit': metadata_for_img['color_digit'],
            'color_bar': metadata_for_img['color_bar'],
            'intervention': metadata_for_img['intervention'],
            'target_tensor': target
        }

        return img, meta_dict

    @property
    def meta_label_strs(self):
        return ["digit", "color_digit", "color_bar", "distr_idx"]

    @property
    def digit_idx(self):
        return 0

    @property
    def color_digit_idx(self):
        return 1

    @property
    def color_bar_idx(self):
        return 2

    @property
    def digit(self):
        return self.labels[:, 0]

    @property
    def color_digit(self):
        return self.labels[:, 1]

    @property
    def color_bar(self):
        return self.labels[:, 2]

    @property
    def latent_dim(self):
        return len(self.meta_label_strs)

    @property
    def distribution_idx(self):
        return self.labels[:, 3]
