from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset


# Define the dataset loader for digit dataset
class CausalCelebA(Dataset):
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

        # load file-list
        self.file_list = []

        # load attrs
        distr_types = ["obs", "int_hair_0", "int_hair_1"]
        self.causal_distr_dfs = dict()
        self.distr_dfs = dict()
        self.causal_main_df = pd.DataFrame()
        for distr_type in distr_types:
            distr_root = root / self.__class__.__name__ / graph_type / distr_type
            filename = distr_root / "causal_attrs.csv"
            attrs_df = pd.read_csv(filename)

            filename = distr_root / "meta_attrs.csv"
            meta_attrs_df = pd.read_csv(filename)

            self.causal_distr_dfs[distr_type] = attrs_df
            self.distr_dfs[distr_type] = meta_attrs_df

            # Find all files matching the pattern
            img_files = list(
                distr_root.rglob("sample_*.jpg")
            )  # Use glob for the current directory, rglob for recursive search
            self.file_list.extend(img_files)

            attrs_df["distr_idx"] = distr_types.index(distr_type)
            attrs_df["distr_type"] = distr_type
            self.causal_main_df = pd.concat(
                [self.causal_main_df, attrs_df], axis=0, ignore_index=True
            )

        if fast_dev_run:
            subsample = 100
            self.causal_main_df = self.causal_main_df.iloc[:subsample]
            self.file_list = self.file_list[:subsample]

        self._load_intervention_targets()

        # XXX: strings do not work nicely with torch
        self.causal_main_df.drop(columns=["Intervention"], inplace=True)

    def _load_intervention_targets(self):
        self.intervention_targets = torch.zeros((len(self), self.latent_dim))

        interv_map = {
            'Haircolor': [2],
        }
        for interv_type in self.causal_main_df['Intervention'].unique():
            idx = self.causal_main_df['Intervention'] == interv_type
            # skip observational as that is all 0's
            if interv_type not in interv_map:
                continue
            
            print(f'Processing intervention type: {interv_type}')
            # set the column to 1, where we intervene
            self.intervention_targets[idx, interv_map[interv_type]] = 1

    @property
    def intervention_targets_per_distr(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]

    @property
    def obs_attr(self):
        return self.causal_distr_dfs["obs"]

    def __len__(self):
        return len(self.causal_main_df)

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

        meta_label = self.causal_main_df.iloc[index]

        # XXX: only can handle one type of intervention
        target = self.intervention_targets[index]
        # print(meta_label)
        # print(target)

        # only extract the array
        meta_label = meta_label.values.tolist()

        if self.transform is not None:
            img = self.transform(img)

        return img, meta_label, target

    @property
    def meta_label_strs(self):
        return [
            "sample_index",
            "gender",
            "age",
            "haircolor",
            "distr_idx",
            "distr_type",
        ]

    @property
    def gender_idx(self):
        return 1

    @property
    def age_idx(self):
        return 2

    @property
    def haircolor_idx(self):
        return 3

    @property
    def latent_dim(self):
        return 3

    @property
    def distribution_idx(self):
        return torch.Tensor(self.causal_main_df['distr_idx'].values)

