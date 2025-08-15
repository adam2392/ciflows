from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from joblib import Parallel, delayed
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from ciflows.datasets.causalmnist_scm.scm import (
    alter_digitbar_img,
    causal_mnist_scm,
)


def log_scm_visualizations(scm_causal_labels, fpath):
    sns.set_context("paper", font_scale=2.0)

    # Plotting the distributions
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=False, sharey=False)
    row_labels = ["A", "B", "C", "D"]
    for idx in range(4):
        causal_labels = scm_causal_labels[idx]
        if idx == 0:
            setting = "Obs. Domain $\Pi_1$"
        elif idx == 1:
            setting = "Obs.  Domain $\Pi_2$"
        elif idx == 2:
            setting = "Soft Int. Domain $\Pi_2$:\n color of bar"
        elif idx == 3:
            # setting = "Hard Int. Domain $\Pi_1$:\n color of digit"
            setting = "Soft Int. Domain $\Pi_1$:\n color of bar"
        # axes[idx, 0].set_ylabel(f"{setting}")
        axes[idx, 0].set_ylabel(f"{setting}")

        digit = causal_labels["digit"]
        color_digit = causal_labels["color_digit"]
        color_bar = causal_labels["color_bar"]

        axes[idx, 0].hist(digit, bins=30, color="black", alpha=0.7)
        axes[idx, 0].set_title("Digit")
        axes[idx, 0].set_xticks(range(10))
        axes[idx, 1].hist(color_digit, bins=30, color="black", alpha=0.7)
        axes[idx, 1].set_title("Color-digit")
        axes[idx, 1].set_xlim([0, 1])
        axes[idx, 1].set_xticks([0, 0.5, 1])
        axes[idx, 2].hist(color_bar, bins=30, color="black", alpha=0.7)
        axes[idx, 2].set_title("Color-bar")
        axes[idx, 2].set_xlim([0, 1])
        axes[idx, 2].set_xticks([0, 0.5, 1])

        # Set the letter as the title of the [idx, 0] figure and position it to the top-right
        axes[idx, 0].set_title(row_labels[idx], loc="left", fontsize=20, fontweight="bold")

    plt.tight_layout()
    # plt.show()
    fig.savefig(fpath, bbox_inches="tight")

    # Parallel image saving helper


def process_and_save_image(idx, img, color_digit, color_bar, save_dir, cmap):
    img_path = save_dir / f"img_{idx}.png"
    new_img = alter_digitbar_img(img, color_digit, color_bar, cmap=cmap, dtype="PIL")
    new_img.save(img_path)
    return img_path


if __name__ == "__main__":
    # where is the data to be saved?
    root = Path("/Users/adam2392/pytorch_data/")
    root = Path("/local/eb/adam2392/")
    dataset_name = "CausalMNIST_v2"
    img_size = 32
    graph_type = "chain_style"
    DEBUG = True

    # set up transforms for each image to augment the dataset
    transform = torchvision.transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),  # Resize images to 128x128
            transforms.CenterCrop(img_size),  # Ensure square crop
            torchvision.transforms.ToTensor(),
        ]
    )
    mnist_data = MNIST(root, train=True, download=True, transform=transform)

    images = mnist_data.data
    labels = mnist_data.targets

    # generate a dictionary of causal labels for each SCM
    scm_causal_labels = dict()
    for intervention_idx in range(4):
        causal_labels = causal_mnist_scm(intervention_idx, labels, graph_type=graph_type)

        print([len(val) for val in causal_labels.values()])
        scm_causal_labels[intervention_idx] = causal_labels

    # log the SCM visualizations
    (root / dataset_name).mkdir(exist_ok=True, parents=True)
    log_scm_visualizations(
        scm_causal_labels, fpath=root / dataset_name / "mnist-colorbar-distributions.pdf"
    )

    target_name_map = {
        0: "observational",
        1: "color_bar",
        2: "color_bar",
        3: "color_digit",
    }
    save_dir_name = {
        0: "observational",
        1: "int_colorbar_0",
        2: "int_colorbar_1",
        3: "int_colorbar_2",
    }

    # save each dataset
    for intervention_idx in [0, 1, 2, 
                            #  3
                             ]:
        distr_folder = root / dataset_name / save_dir_name[intervention_idx]
        img_distr_folder = distr_folder / "images"
        img_distr_folder.mkdir(exist_ok=True, parents=True)

        causal_labels = scm_causal_labels[intervention_idx]
        # causal_labels["distr_idx"] = torch.tensor([intervention_idx] * len(labels))
        intervention_target_tensor = torch.tensor(
            causal_labels["intervention_targets"], dtype=torch.int
        )
        # intervention_target_tensor = torch.zeros((len(causal_labels), 3), dtype=torch.int)

        # extract the meta-data from the causal labels
        keys = ["digit", "color_digit", "color_bar"]
        label_tensor = torch.zeros((len(intervention_target_tensor), len(keys)))
        # convert the labels from a list of dictionaries to a tensor array
        for idx, key in enumerate(keys):
            label_tensor[:, idx] = torch.tensor(causal_labels[key])
        distr_idx = torch.tensor(
            [intervention_idx] * len(intervention_target_tensor), dtype=torch.int
        )
        label_tensor = torch.hstack([label_tensor, distr_idx.reshape(-1, 1)])
        labels_df = pd.DataFrame(label_tensor, columns=keys + ["distr_idx"])
        labels_df["intervention"] = target_name_map[intervention_idx]
        labels_df[["digit", "distr_idx"]] = labels_df[["digit", "distr_idx"]].astype(int)

        # save the images
        all_labels = []
        # Set colormaps once per intervention
        color_digit_cmap = causal_labels.get("color_digit_cmap", "gist_rainbow")
        color_bar_cmap = causal_labels.get("color_bar_cmap", "gist_rainbow")

        # Parallel image saving
        img_paths = Parallel(n_jobs=1)(
            delayed(process_and_save_image)(
                idx,
                img,
                causal_labels["color_digit"][idx],
                causal_labels["color_bar"][idx],
                img_distr_folder,
                cmap=color_digit_cmap,
            )
            for idx, img in tqdm(list(enumerate(images)), desc="Saving images")
        )

        # Set DataFrame metadata
        labels_df["color_digit_cmap"] = color_digit_cmap
        labels_df["color_bar_cmap"] = color_bar_cmap
        labels_df["color_digit"] = labels_df["color_digit"].astype(int)
        labels_df["color_bar"] = labels_df["color_bar"].astype(int)

        # If you want to also store image filenames
        labels_df["img_path"] = [str(p.relative_to(img_distr_folder.parent)) for p in img_paths]

        print(intervention_target_tensor.shape)
        print(labels_df.shape)
        print(len(images))

        # save the intervention targets
        targets_fname = distr_folder / "targets.pt"
        torch.save(intervention_target_tensor, targets_fname)

        # Save the concatenated causal labels for meta-analysis in postprocessing
        # all_labels_df = pd.concat(all_labels, ignore_index=True)
        # overall_meta_path = distr_folder / "all_causal_attrs.csv"
        # all_labels_df.to_csv(overall_meta_path, index=False)
        # Save per-distribution CSV
        meta_path = distr_folder / "causal_labels.csv"
        labels_df.to_csv(meta_path, index=False)
        print(f"Saved overall causal attribution CSV to: {meta_path}")

        print(intervention_target_tensor.shape)
        print(label_tensor.shape)
        # print(all_labels_df.head())
