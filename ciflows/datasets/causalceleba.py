import re
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
        input_transform=None,
        img_size=64,
        target_transform=None,
        fast_dev_run=False,
    ):
        self.root = root
        self.transform = transform
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.graph_type = graph_type
        self.img_size = img_size

        root = Path(root)

        # load file-list
        self.file_list = []

        # load attrs
        distr_types = ["obs", "int_hair_0", "int_hair_1", "int_hair_2"]  # , "int_hair_3"]
        self.causal_distr_dfs = dict()
        self.distr_dfs = dict()
        self.causal_main_df = pd.DataFrame()
        for distr_type in distr_types:
            distr_root = root / self.__class__.__name__ / graph_type / f"dim{img_size}" / distr_type
            filename = distr_root / "causal_attrs.csv"
            attrs_df = pd.read_csv(filename, index_col=0)

            self.causal_distr_dfs[distr_type] = attrs_df

            # Find all files matching the pattern
            img_files = list(
                distr_root.rglob("sample_*.jpg")
            )  # Use glob for the current directory, rglob for recursive search
            # Sort the images based on the number after "sample_"
            img_files_sorted = sorted(
                img_files, key=lambda x: int(re.search(r"sample_(\d+)", x.name).group(1))
            )

            self.file_list.extend(img_files_sorted)

            attrs_df["distr_type"] = distr_type
            attrs_df["distr_idx"] = distr_types.index(distr_type)
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
            "Haircolor": [2],
        }
        for interv_type in self.causal_main_df["Intervention"].unique():
            idx = self.causal_main_df["Intervention"] == interv_type
            # skip observational as that is all 0's
            if interv_type not in interv_map:
                continue

            print(f"Processing intervention type: {interv_type}")
            # set the column to 1, where we intervene
            self.intervention_targets[idx, interv_map[interv_type]] = 1

    @property
    def intervention_targets_per_distr(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            # [0, 0, 1],
        ]

    @property
    def distr_idx_list(self):
        return [0, 1, 2, 3]

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
        distr_idx = meta_label["distr_idx"]

        # XXX: only can handle one type of intervention
        target = self.intervention_targets[index]
        # print(meta_label)
        # print(target)

        # only extract the array
        meta_label = meta_label.values.tolist()

        if self.transform is not None:
            img = self.transform(img)

        if self.input_transform is not None:
            image = torch.permute(img, (1, 2, 0))
            # print(image.shape)
            # img = torch.permute(img, (1, 2, 0))
            input_img = self.input_transform(image=np.array(image))["image"]
        else:
            input_img = img

        return input_img, img, distr_idx, target, meta_label

    @property
    def meta_label_strs(self):
        return [
            "sample_index",
            "gender",
            "age",
            "haircolor",
            "distr_type",
            "distr_idx",
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
        return torch.Tensor(self.causal_main_df["distr_idx"].values)

    def sample(self, n_samples, attr_name, attr_val, seed=None):
        """Sample n_samples of the given attribute value.

        For example, one can sample 10 samples of images with
        haircolor = 2, corresponding to brown hair.

        Parameters
        ----------
        n_samples : int
            Number of samples to sample.
        attr_name : str
            Attribute name to sample. Should be a column in the `causal_main_df`.
        attr_val : int or float
            Attribute value to sample. Should be a value in the `causal_main_df[attr_name]`.

        Returns
        -------
        samples : array-like of shape (n_samples, ...)
            Samples of the given attribute value.
        idx : np.ndarray
            Index of the samples in the dataset.
        """
        rng = np.random.default_rng(seed)
        idx = self.causal_main_df[attr_name] == attr_val
        idx = idx.to_numpy().nonzero()[0]

        if len(idx) < n_samples:
            print(idx)
            raise ValueError(f"Cannot sample {n_samples} samples with {attr_name} = {attr_val}")

        # randomly sample n_samples from idx
        rand_idx = rng.choice(idx, n_samples, replace=False)
        # print(idx)
        samples = []
        for i in rand_idx:
            img, _, distr_idx, target, meta_label = self[i]
            samples.append(img)
        samples = torch.stack(samples, dim=0)
        return samples, rand_idx


class CausalCelebAEyeGlasses(Dataset):
    def __init__(
        self,
        root,
        graph_type,
        transform=None,
        img_size=64,
        target_transform=None,
        fast_dev_run=False,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.graph_type = graph_type
        self.img_size = img_size

        root = Path(root)

        # load file-list
        self.file_list = []

        # load attrs
        distr_types = ["obs", "int_eye_0", "int_eye_1"]
        self.causal_distr_dfs = dict()
        self.distr_dfs = dict()
        self.causal_main_df = pd.DataFrame()
        for distr_type in distr_types:
            distr_root = root / self.__class__.__name__ / graph_type / f"dim{img_size}" / distr_type
            filename = distr_root / "causal_attrs.csv"
            attrs_df = pd.read_csv(filename)

            self.causal_distr_dfs[distr_type] = attrs_df

            # Find all files matching the pattern
            img_files = list(
                distr_root.rglob("sample_*.jpg")
            )  # Use glob for the current directory, rglob for recursive search
            self.file_list.extend(img_files)

            attrs_df["distr_type"] = distr_type
            attrs_df["distr_idx"] = distr_types.index(distr_type)
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
            "Eyeglasses": [2],
        }
        for interv_type in self.causal_main_df["Intervention"].unique():
            idx = self.causal_main_df["Intervention"] == interv_type
            # skip observational as that is all 0's
            if interv_type not in interv_map:
                continue

            print(f"Processing intervention type: {interv_type}")
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
    def distr_idx_list(self):
        return [0, 1, 2, 3, 4, 5]

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
        - eyeglasses

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
        distr_idx = meta_label["distr_idx"]

        # XXX: only can handle one type of intervention
        target = self.intervention_targets[index]
        # print(meta_label)
        # print(target)

        # only extract the array
        meta_label = meta_label.values.tolist()

        if self.transform is not None:
            img = self.transform(img)

        return img, distr_idx, target, meta_label

    @property
    def meta_label_strs(self):
        return [
            "sample_index",
            "gender",
            "age",
            "eyeglasses",
            "distr_type",
            "distr_idx",
        ]

    @property
    def gender_idx(self):
        return 1

    @property
    def age_idx(self):
        return 2

    @property
    def eyeglasses_idx(self):
        return 3

    @property
    def latent_dim(self):
        return 3

    @property
    def distribution_idx(self):
        return torch.Tensor(self.causal_main_df["distr_idx"].values)


class CausalCelebAEmbedding(CausalCelebA):
    def __init__(
        self,
        root,
        graph_type,
        dataset,
        scm_type,
        img_size=64,
        fast_dev_run=False,
    ):
        self.root = root
        self.graph_type = graph_type
        self.img_size = img_size

        root = Path(root)

        # load attrs
        if dataset == "alldata":
            dataset_postfix = "alldata_encodings"
            encoding_fnames = {
                "obs": f"obs_{dataset_postfix}.pt",
                f"int_{scm_type}_0": f"int_{scm_type}_0_{dataset_postfix}.pt",
                f"int_{scm_type}_1": f"int_{scm_type}_1_{dataset_postfix}.pt",
                f"int_{scm_type}_2": f"int_{scm_type}_2_{dataset_postfix}.pt",
                # f"int_{scm_type}_3": f"int_{scm_type}_3_{dataset_postfix}.pt",
                # "int_hair_4": f"int_hair_4_{dataset_postfix}.pt",
                # "obs": "obs_nonorm_encodings.pt",
                # "int_hair_0": "int_hair_0_nonorm_encodings.pt",
                # "int_hair_1": "int_hair_1_nonorm_encodings.pt",
            }
        elif dataset == "alldata_l1loss":
            dataset_postfix = "alldata_l1loss_encodings"
            encoding_fnames = {
                "obs": f"obs_{dataset_postfix}.pt",
                f"int_{scm_type}_0": f"int_{scm_type}_0_{dataset_postfix}.pt",
                f"int_{scm_type}_1": f"int_{scm_type}_1_{dataset_postfix}.pt",
                f"int_{scm_type}_2": f"int_{scm_type}_2_{dataset_postfix}.pt",
                # f"int_{scm_type}_3": f"int_{scm_type}_3_{dataset_postfix}.pt",
                # "int_hair_4": f"int_hair_4_{dataset_postfix}.pt",
                # "obs": "obs_nonorm_encodings.pt",
                # "int_hair_0": "int_hair_0_nonorm_encodings.pt",
                # "int_hair_1": "int_hair_1_nonorm_encodings.pt",
            }
        elif dataset == "cyclicbeta":
            dataset_postfix = "cyclicbeta_noimgaug_encodings"
            encoding_fnames = {
                "obs": f"obs_{dataset_postfix}.pt",
                "int_hair_0": f"int_hair_0_{dataset_postfix}.pt",
                "int_hair_1": f"int_hair_1_{dataset_postfix}.pt",
                "int_hair_2": f"int_hair_2_{dataset_postfix}.pt",
                "int_hair_3": f"int_hair_3_{dataset_postfix}.pt",
                # "int_hair_4": f"int_hair_4_{dataset_postfix}.pt",
                # "obs": "obs_nonorm_encodings.pt",
                # "int_hair_0": "int_hair_0_nonorm_encodings.pt",
                # "int_hair_1": "int_hair_1_nonorm_encodings.pt",
            }

        print()
        print()
        print(f"Loaded dataset postfix: {dataset_postfix}")
        distr_types_list = [
            "obs",
            f"int_{scm_type}_0",
            f"int_{scm_type}_1",
            f"int_{scm_type}_2",
            # "int_hair_3",
            # "int_hair_4",
        ]
        self.causal_distr_dfs = dict()
        self.causal_main_df = pd.DataFrame()
        self.data = []
        for distr_type in distr_types_list:
            distr_root = root / "CausalCelebA" / graph_type / f"dim{img_size}" / distr_type
            # distr_root = (
            #     root / "CausalCelebAEyeGlasses" / graph_type / f"dim{img_size}" / distr_type
            # )
            filename = distr_root / "causal_attrs.csv"
            attrs_df = pd.read_csv(filename)

            self.causal_distr_dfs[distr_type] = attrs_df

            attrs_df["distr_idx"] = distr_types_list.index(distr_type)
            attrs_df["distr_type"] = distr_type
            self.causal_main_df = pd.concat(
                [self.causal_main_df, attrs_df], axis=0, ignore_index=True
            )

            # Find data files
            encoding_fname = encoding_fnames[distr_type]
            self.data.append(torch.load(distr_root / encoding_fname))

        # concatenate all data as a single Tensor
        self.data = torch.cat(self.data, dim=0)

        if fast_dev_run:
            subsample = 100
            self.causal_main_df = self.causal_main_df.iloc[:subsample]
            self.file_list = self.file_list[:subsample]

        self._load_intervention_targets()

        # XXX: strings do not work nicely with torch
        self.causal_main_df.drop(columns=["Intervention"], inplace=True)

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
        img = self.data[index].squeeze()

        meta_label = self.causal_main_df.iloc[index]
        distr_idx = meta_label["distr_idx"]
        # XXX: only can handle one type of intervention
        target = self.intervention_targets[index]
        # print(meta_label)
        # print(target)

        # only extract the array
        # print(meta_label)
        meta_label = meta_label.values.tolist()

        return img, distr_idx, target, meta_label
