from pathlib import Path

import PIL
import torch
from torch.utils.data import Dataset


# Define the dataset loader for digit dataset
class CausalDigitBarMNIST(Dataset):
    def __init__(
        self,
        root,
        graph_type,
        transform=None,
        target_transform=None,
        fast_dev_run=False,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.graph_type = graph_type

        root = Path(root)

        # load data from disc
        self.data = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-imgs-train.pt"
        )
        self.labels = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-labels-train.pt"
        )
        if isinstance(self.labels, list):
            self.labels = torch.vstack(self.labels)

        self.intervention_targets = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-targets-train.pt"
        )
        if isinstance(self.intervention_targets, list):
            self.intervention_targets = torch.vstack(self.intervention_targets)

        if not all(
            [
                len(self.data) == len(self.labels),
                len(self.data) == len(self.intervention_targets),
            ]
        ):
            raise ValueError("Data, labels and intervention targets must have the same length.")

        if fast_dev_run:
            subsample = 100
            self.data = self.data[:subsample]
            self.labels = self.labels[:subsample]
            self.intervention_targets = self.intervention_targets[:subsample]

    @property
    def intervention_targets_per_distr(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample from the image dataset.

        The target composes of the meta-labeling:
        - width
        - color
        - fracture_thickness
        - fracture_num_fractures
        - label
        """
        img, meta_label, target = (
            self.data[index],
            self.labels[index],
            self.intervention_targets[index],
        )

        # get the distribution index
        distr_idx = meta_label[-1]

        if self.transform is not None:
            img = self.transform(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = PIL.Image.fromarray(img.numpy(), mode="RGB")
        return img, distr_idx, target, meta_label

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
        return self.labels.shape[1]

    @property
    def distribution_idx(self):
        return self.labels[:, 3]


class CausalMNISTEmbedding(CausalDigitBarMNIST):
    def __init__(
        self,
        root,
        graph_type,
        transform=None,
        target_transform=None,
        fast_dev_run=False,
    ):
        self.root = root
        self.graph_type = graph_type

        root = Path(root)

        # load attrs
        dataset = 'alldata'
        if dataset == "alldata":
            dataset_postfix = "alldata_encodings"
            fname = f"{graph_type}_{dataset_postfix}.pt"

        print()
        print()
        print(f"Loaded dataset postfix: {dataset_postfix}")
        self.data = torch.load(fname)

        self.labels = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-labels-train.pt"
        )
        if isinstance(self.labels, list):
            self.labels = torch.vstack(self.labels)

        self.intervention_targets = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-targets-train.pt"
        )
        if isinstance(self.intervention_targets, list):
            self.intervention_targets = torch.vstack(self.intervention_targets)

        if not all(
            [
                len(self.data) == len(self.labels),
                len(self.data) == len(self.intervention_targets),
            ]
        ):
            raise ValueError("Data, labels and intervention targets must have the same length.")

        

        if fast_dev_run:
            subsample = 100
            self.causal_main_df = self.causal_main_df.iloc[:subsample]
            self.file_list = self.file_list[:subsample]


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
        distr_idx : int
            Which distribution associated.
        target : torch.Tensor of shape (latent_dim,)
            Intervention target with 1's where the intervention is applied.
        meta_label : list
            List of meta-labels
        """
        img, meta_label, target = (
            self.data[index],
            self.labels[index],
            self.intervention_targets[index],
        )

        # get the distribution index
        distr_idx = meta_label[-1]

        if self.transform is not None:
            img = self.transform(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = PIL.Image.fromarray(img.numpy(), mode="RGB")
        return img, distr_idx, target, meta_label
