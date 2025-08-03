import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from PIL import Image
from scipy.stats import truncnorm

from ciflows.datasets.causalmnist_scm.utils import value_to_rgb

from .perturb import Fracture, Thickening, Thinning, apply_perturbation


def add_bar(img, color_bar_val=255, start_height=0, height=3, start_width=0, width=None):
    if width is None:
        width = img.shape[1]

    # add bar
    for color_idx in range(3):
        img[
            start_height : start_height + height,
            start_width : start_width + width,
            color_idx,
        ] = (
            color_bar_val[color_idx] * 255.0
        )
    return img


def truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std
    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    return torch.tensor(samples, dtype=torch.float32)


def skewed_normal(mode, skewness, size):
    alpha = skewness
    loc = mode
    scale = 0.1  # Adjust scale as necessary
    samples = scipy.stats.skewnorm.rvs(a=alpha, loc=loc, scale=scale, size=size)
    return torch.tensor(samples, dtype=torch.float32)


######################################################################
# SCMs for the Causal MNIST dataset
######################################################################


def nonmarkov_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar following a nonmarkov structure.

    color_bar <--> color_digit <- digit -> style

    Parameters
    ----------
    intervention_idx : int
        The intervention index.
    labels : tensor of shape (n_samples,)
        The labels of each of the MNIST digit samples. This corresponds to the
        digit. There is an intervention on the color-digit.

    Returns
    -------
    causal_labels : dict
        A dictionary of:
            - digit : tensor of shape (n_samples,)
                The digit label.
            - color_digit : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - color_bar : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
    digit_idx : list of int
        The indices of each image we want in the dataset. This may contain
        replicates of the indices of size (n_samples,).
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    digit_idx = torch.arange(n_samples)
    digit = torch.Tensor(labels)

    color_digit = torch.zeros(n_samples)
    color_bar = torch.zeros(n_samples)

    U_cdigit_cbar = torch.normal(mean=0.5, std=0.2, size=(n_samples, 1)).squeeze()
    color_digit = torch.zeros(n_samples)

    if intervention_idx == 0:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit_means = torch.linspace(
            0, 1, 10
        )  # 10 possible digits, evenly spaced means from 0 to 1
        color_digit_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx == 1:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit_means = torch.linspace(
            0, 2, 10
        )  # 10 possible digits, evenly spaced means from 0 to 1
        color_digit_means = torch.flip(color_digit_means, [0]) / color_digit_means.sum()
        color_digit_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit
        causal_labels["intervention_targets"] = torch.Tensor([[0, 1, 0]] * n_samples)
    elif intervention_idx == 2:
        # Define the skewed probabilities (higher for digits 5-9)
        digit_probabilities = torch.tensor(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 0.2, 0.25, 0.3, 0.35]
        )
        # Normalize the probabilities to sum to 1 (optional but recommended)
        digit_probabilities /= digit_probabilities.sum()

        # probability of sampling each image
        sampling_probabilities = digit_probabilities[labels]

        # Sample with replacement using the skewed probabilities
        digit_idx = torch.multinomial(sampling_probabilities, n_samples, replacement=True)
        # Get the sampled labels
        digit = labels[digit_idx]

        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit_means = torch.linspace(
            0, 1, 10
        )  # 10 possible digits, evenly spaced means from 0 to 1
        color_digit_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit
        causal_labels["intervention_targets"] = torch.Tensor([[1, 0, 0]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    # sample the color-digit conditioned on the digit
    for i in range(5):
        mask = digit == i * 2
        mask_2 = digit == i * 2 + 1
        mask = mask | mask_2
        num_samples = mask.sum().item()
        color_digit[mask] = truncated_normal(
            color_digit_means[i] + U_cdigit_cbar[mask],
            color_digit_stds[i],
            0,
            1,
            num_samples,
        )

    # color-bar is caused by color-digit
    color_bar = truncated_normal(U_cdigit_cbar, 0.1, 0, 1, n_samples)

    # color_bar <--> color_digit <- digit -> style

    causal_labels.update(
        {
            "digit": digit,
            "color_digit": color_digit,
            "color_bar": color_bar,
            "confounder": U_cdigit_cbar,
        }
    )
    return causal_labels, digit_idx


def collider_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar following a collider structure.

    The color-bar will be a function of both the digit and the color-digit.

    Parameters
    ----------
    intervention_idx : int
        The intervention index.
    labels : tensor of shape (n_samples,)
        The labels of each of the MNIST digit samples. This corresponds to the
        digit.

    Returns
    -------
    causal_labels : dict
        A dictionary of:
            - digit : tensor of shape (n_samples,)
                The digit label.
            - color_digit : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - color_bar : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
    digit_idx : list of int
        The indices of each image we want in the dataset. This may contain
        replicates of the indices of size (n_samples,).
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    color_digit = torch.zeros(n_samples)
    color_bar = torch.zeros(n_samples)

    if intervention_idx == 0:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit = torch.normal(
            mean=0.5, std=0.2, size=(n_samples, 1)
        ).squeeze()  # 10 possible digits, evenly spaced means from 0 to 1
        digit = torch.Tensor(labels)
        digit_idx = torch.arange(n_samples)

        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx in (1, 2, 3, 4):
        # Define the skewed probabilities (higher for digits 5-9)
        digit_probabilities = torch.tensor(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35]
        )
        digit_probabilities[:5] *= intervention_idx * 2
        # Normalize the probabilities to sum to 1 (optional but recommended)
        digit_probabilities /= digit_probabilities.sum()

        # probability of sampling each image
        sampling_probabilities = digit_probabilities[labels]

        # Sample with replacement using the skewed probabilities
        digit_idx = torch.multinomial(sampling_probabilities, n_samples, replacement=True)
        # Get the sampled labels
        digit = labels[digit_idx]

        # Intervention 1: mean equal to 1 - color-digit
        # color_bar = truncated_normal(1 - color_digit, 0.1, 0, 1, n_samples)
        # Intervention 2: skewed distribution towards 0.9
        # color_digit = skewed_normal(intervention_idx / 10., 1.0 / (color_digit + 0.5), n_samples)
        color_digit = truncated_normal(intervention_idx / 5.0, 0.2, 0, 1, n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[1, 1, 0]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    # for each digit, condition on the corresponding color-digit and sample the color-bar
    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        color_bar[mask] = truncated_normal(color_digit[i], 0.2, 0, 1, num_samples)

    causal_labels.update({"digit": digit, "color_digit": color_digit, "color_bar": color_bar})

    return causal_labels, digit_idx


def chain_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar.

    Digit -> Color-Digit -> Color-Bar

    There are four possible distributions that are generated. The first is observational
    (intervention_idx == 0):

        - digit is just uniformly distributed more or less in the MNIST dataset
        - color-digit will be a mixture of gaussians, where the mean is evenly
        spaced from 0 to 1 depending on the digit, and the standard deviation
        is a default value of 0.1 for each digit
        - color-bar will be a function of color-digit, sampled from a truncated
        normal distribution with mean equal to the color-digit value and a standard
        deviation of 0.1

    Intervention 1 (color-bar) is changed:
        - color-bar will be sampled from a truncated normal distribution with
        mean equal to 1 - color-digit and a standard deviation of 0.1

    Intervention 2 (color-bar) is changed:
        - color-bar will be sampled from a skewed distribution towards
        0.9.

    Parameters
    ----------
    intervention_idx : int
        The intervention index.
    labels : tensor of shape (n_samples,)
        The labels of each of the MNIST digit samples. This corresponds to the
        digit.

    Returns
    -------
    causal_labels : dict
        A dictionary of:
            - digit : tensor of shape (n_samples,)
                The digit label.
            - color_digit : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - color_bar : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - intervention_targets : tensor of shape (n_samples, 3)
                The intervention targets for the color-bar. This is a one-hot
                encoding of the intervention index, where a one indicates
                which causal variable is intervened on.
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    digit = torch.Tensor(labels)

    # Color-digit will be a mixture of gaussians
    color_digit_means = torch.linspace(
        0, 1, 10
    )  # 10 possible digits, evenly spaced means from 0 to 1
    color_digit_stds = 0.05 * torch.ones(10)  # Standard deviation of 0.15 for each digit
    color_digit = torch.zeros(n_samples)

    # sample the color-digit conditioned on the digit
    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        # color_digit[mask] = torch.distributions.Normal(
        #     color_digit_means[i], color_digit_stds[i]
        # ).sample((num_samples,))
        color_digit[mask] = truncated_normal(
            color_digit_means[i], color_digit_stds[i], 0, 1, num_samples
        )
    snr = 0.2

    # Generate color_bar based on the intervention index
    if intervention_idx == 0:
        # Observational distribution
        # color_bar = truncated_normal(0.5, 0.2 / (color_digit + 1), 0, 1, n_samples)
        color_bar = truncated_normal(1.0 / (color_digit + 1), 0.1, 0, 1, n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx == 1:
        # Intervention 1: mean equal to 1 - color-digit
        # color_bar = truncated_normal(1 - color_digit, 0.1, 0, 1, n_samples)
        # Intervention 2: skewed distribution towards 0.9
        color_bar = skewed_normal(snr, 1.0 / (color_digit + 0.5), n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    elif intervention_idx == 2:
        # Intervention 2: skewed distribution towards 0.9
        color_bar = skewed_normal(1.0 - snr, 1.0 / (color_digit + 0.5), n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    elif intervention_idx == 3:
        # Intervention 3: hard normal distribution on color_digit
        n_samples = labels.shape[0]
        color_digit = truncated_normal(0.5, 0.2, 0, 1, n_samples)
        # color_bar = truncated_normal(0.5, 0.2 / (color_digit + 0.5), 0, 1, n_samples)
        color_bar = truncated_normal(0.75 / (color_digit + 1), 0.1, 0, 1, n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 1, 0]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    causal_labels.update({"digit": digit, "color_digit": color_digit, "color_bar": color_bar})

    return causal_labels


def chain_style_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar.

    style <- Digit -> Color-Digit -> Color-Bar

    There are four possible distributions that are generated. The first is observational
    (intervention_idx == 0):

        - digit is just uniformly distributed more or less in the MNIST dataset
        - color-digit will be a mixture of gaussians, where the mean is evenly
        spaced from 0 to 1 depending on the digit, and the standard deviation
        is a default value of 0.1 for each digit
        - color-bar will be a function of color-digit, sampled from a truncated
        normal distribution with mean equal to the color-digit value and a standard
        deviation of 0.1

    Intervention 1 (color-bar) is changed:
        - color-bar will be sampled from a truncated normal distribution with
        mean equal to 1 - color-digit and a standard deviation of 0.1

    Intervention 2 (color-bar) is changed:
        - color-bar will be sampled from a skewed distribution towards
        0.9.

    Parameters
    ----------
    intervention_idx : int
        The intervention index.
    labels : tensor of shape (n_samples,)
        The labels of each of the MNIST digit samples. This corresponds to the
        digit.

    Returns
    -------
    causal_labels : dict
        A dictionary of:
            - digit : tensor of shape (n_samples,)
                The digit label.
            - color_digit : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - color_bar : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - intervention_targets : tensor of shape (n_samples, 3)
                The intervention targets for the color-bar. This is a one-hot
                encoding of the intervention index, where a one indicates
                which causal variable is intervened on.
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    digit = torch.Tensor(labels)

    # Color-digit will be a mixture of gaussians
    color_digit_means = torch.linspace(
        0, 1, 10
    )  # 10 possible digits, evenly spaced means from 0 to 1
    color_digit_stds = 0.05 * torch.ones(10)  # Standard deviation of 0.15 for each digit
    color_digit = torch.zeros(n_samples)
    width = torch.zeros(n_samples)
    width_means = torch.linspace(-0.75, 0.75, 10)
    # sample the color-digit conditioned on the digit

    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        # color_digit[mask] = torch.distributions.Normal(
        #     color_digit_means[i], color_digit_stds[i]
        # ).sample((num_samples,))
        color_digit[mask] = truncated_normal(
            color_digit_means[i], color_digit_stds[i], 0, 1, num_samples
        )

        width[mask] = torch.normal(width_means[i], 0.05, size=(num_samples,))

    snr = 0.2

    # Generate color_bar based on the intervention index
    if intervention_idx == 0:
        # Observational distribution
        # color_bar = truncated_normal(0.5, 0.2 / (color_digit + 1), 0, 1, n_samples)
        color_bar = truncated_normal(1.0 / (color_digit + 1), 0.1, 0, 1, n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx == 1:
        # Intervention 1: mean equal to 1 - color-digit
        # color_bar = truncated_normal(1 - color_digit, 0.1, 0, 1, n_samples)
        # Intervention 2: skewed distribution towards 0.9
        color_bar = skewed_normal(snr, 1.0 / (color_digit + 0.5), n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    elif intervention_idx == 2:
        # Intervention 2: skewed distribution towards 0.9
        color_bar = skewed_normal(1.0 - snr, 1.0 / (color_digit + 0.5), n_samples)
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    elif intervention_idx == 3:
        # Intervention 3: hard normal distribution on color_digit
        # n_samples = labels.shape[0]
        # color_digit = truncated_normal(0.5, 0.2, 0, 1, n_samples)
        # # color_bar = truncated_normal(0.5, 0.2 / (color_digit + 0.5), 0, 1, n_samples)
        color_bar = truncated_normal(0.75 / (color_digit + 1), 0.1, 0, 1, n_samples)
        width = torch.rand(size=(n_samples,)) - 0.5
        # for i in range(10):
        #     mask = digit == i
        #     num_samples = mask.sum().item()
        #     width[mask] = torch.normal(width_means[-i], 0.1, size=(num_samples,))
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 1]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    causal_labels.update(
        {"digit": digit, "color_digit": color_digit, "color_bar": color_bar, "width": width}
    )

    return causal_labels


# Placeholder for alter_img function
# def alter_digitbar_img(img, color_digit, color_bar):
#     cmap = plt.cm.viridis

#     # change the color
#     h, w = img.shape
#     color_value = value_to_rgb(color_digit)
#     colored_arr = np.zeros((h, w, 3), dtype=np.uint8)
#     mask = img > 0  # Mask to identify the digit
#     for i in range(3):
#         colored_arr[:, :, i][mask] = color_value[i] * 255  # Apply uniform color to the digit
#     img = colored_arr

#     # add bar
#     # color_value = np.array(cmap(color_digit.item())[:3]).squeeze()
#     color_value = value_to_rgb(color_bar, methods="cmap", cmap=cmap)
#     img = add_bar(
#         img,
#         color_bar_val=color_value,
#         start_height=0,
#         height=4,
#         start_width=0,
#         width=None,
#     )
#     return img


def causal_mnist_scm(intervention_idx, labels, graph_type="chain"):
    if graph_type == "chain":
        causal_labels = chain_scm(intervention_idx=intervention_idx, labels=labels)
    elif graph_type == "collider":
        causal_labels = collider_scm(intervention_idx=idx, labels=labels)
    elif graph_type == "nonmarkov":
        causal_labels = nonmarkov_scm(intervention_idx=idx, labels=labels)
    elif graph_type == "chain_style":
        causal_labels = chain_style_scm(intervention_idx=intervention_idx, labels=labels)
    else:
        raise ValueError("Invalid graph_type. Must be chain, collider or nonmarkov.")
    return causal_labels


# Placeholder for alter_img function
def alter_digitbar_img(img, color_digit, color_bar, cmap=None, dtype=None):
    """Alter a MNIST image by changing the color of the digit and adding a color bar.

    Parameters
    ----------
    img : Image
        The MNIST image.
    color_digit : float
        The color of the digit in [0, 1].
    color_bar : float
        The color of the bar in [0, 1].

    Returns
    -------
    img : Image as numpy array, PIL.Image or torch.Tensor of shape (28, 28, 3)
        The output image.
    """
    if cmap is None:
        cmap = plt.cm.viridis
    else:
        cmap = plt.get_cmap(cmap)

    # change the color
    h, w = img.shape
    color_value = value_to_rgb(color_digit, methods="cmap", cmap=cmap)
    colored_arr = np.zeros((h, w, 3), dtype=np.uint8)
    mask = img > 0  # Mask to identify the digit
    for i in range(3):
        colored_arr[:, :, i][mask] = color_value[i] * 255  # Apply uniform color to the digit
    img = colored_arr

    # add bar
    # color_value = np.array(cmap(color_digit.item())[:3]).squeeze()
    color_value = value_to_rgb(color_bar, methods="cmap", cmap=cmap)
    img = add_bar(
        img,
        color_bar_val=color_value,
        start_height=0,
        height=4,
        start_width=0,
        width=None,
    )

    if dtype == "PIL":
        img = Image.fromarray(img, mode="RGB")
    elif dtype == "torch":
        img = torch.tensor(img, dtype=torch.float32)
    elif dtype is not None:
        raise ValueError(f"Invalid dtype: {dtype} Must be PIL or torch or None for numpy array.")
    return img


def alter_digitbar_style_img(
    img, width, color_digit, color_bar, cmap=None, dtype=None, apply_fracture=False
):
    """Alter a MNIST image by changing the color of the digit and adding a color bar.

    Parameters
    ----------
    img : Image
        The MNIST image.
    color_digit : float
        The color of the digit in [0, 1].
    color_bar : float
        The color of the bar in [0, 1].
    cmap : str, optional
        The colormap for the colors to use.
    apply_fracture : bool
        Whether to apply a random fracture to the image

    Returns
    -------
    img : Image as numpy array, PIL.Image or torch.Tensor of shape (28, 28, 3)
        The output image.
    """
    if cmap is None:
        cmap = plt.cm.viridis
    else:
        cmap = plt.get_cmap(cmap)

    # change the width
    # Apply transformations to the image based on the parameters
    # This is just a placeholder, you need to implement the actual transformation logic
    # apply width changes
    if width < 0.0:
        width = np.abs(width)
        width_func = Thinning(amount=width)
    else:
        width_func = Thickening(amount=width)

    img = img.squeeze()
    img = apply_perturbation(img, perturbation=width_func, convert_dtype=False)
    img = torch.tensor(img)

    if apply_fracture:
        thickness = max(1.1, 1.0 - width)
        fracture_func = Fracture(thickness=thickness, prune=1, num_frac=1)
        img = apply_perturbation(img, perturbation=fracture_func, convert_dtype=False)

    # Define a function for random rotation
    from skimage.transform import rotate

    def random_rotation(image, max_angle=20):
        angle = np.random.uniform(-max_angle, max_angle)
        rotated_image = rotate(image, angle, resize=False)
        return rotated_image

    # Apply random rotation
    img = random_rotation(img)

    # change the color
    h, w = img.shape
    color_value = value_to_rgb(color_digit, methods="cmap", cmap=cmap)
    colored_arr = np.zeros((h, w, 3), dtype=np.uint8)
    mask = img > 0  # Mask to identify the digit
    for i in range(3):
        colored_arr[:, :, i][mask] = color_value[i] * 255  # Apply uniform color to the digit
    img = colored_arr

    # add bar
    # color_value = np.array(cmap(color_digit.item())[:3]).squeeze()
    color_value = value_to_rgb(color_bar, methods="cmap", cmap=cmap)
    img = add_bar(
        img,
        color_bar_val=color_value,
        start_height=0,
        height=4,
        start_width=0,
        width=None,
    )

    if dtype == "PIL":
        img = Image.fromarray(img, mode="RGB")
    elif dtype == "torch":
        img = torch.tensor(img, dtype=torch.float32)
    elif dtype is not None:
        raise ValueError(f"Invalid dtype: {dtype} Must be PIL or torch or None for numpy array.")
    return img


if __name__ == "__main__":
    from pathlib import Path

    import torchvision
    from torchvision.datasets import MNIST

    img_dataset = []
    labels_dataset = None

    root = "/Users/adam2392/pytorch_data/"

    # set up transforms for each image to augment the dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # nf.utils.Scale(255.0 / 256.0),  # normalize the pixel values
            # nf.utils.Jitter(1 / 256.0),  # apply random generation
            # torchvision.transforms.RandomRotation(350),  # get random rotations
        ]
    )

    mnist_data = MNIST(root, train=True, download=True, transform=transform)

    images = mnist_data.data
    labels = mnist_data.targets

    print(len(images))
    for intervention_idx in [0, 1, 2, 3]:
        causal_labels = bar_digit_scm(intervention_idx=intervention_idx, labels=labels)
        causal_labels["distr_idx"] = torch.Tensor([intervention_idx] * len(labels))

        # ensure all tensors are 2D so vstackable
        for key in keys:
            if causal_labels[key].ndim == 1:
                causal_labels[key] = causal_labels[key].reshape(-1, 1)

        if labels_dataset is None:
            labels_dataset = causal_labels
        else:
            keys = list(causal_labels.keys())
            for key in keys:
                labels_dataset[key] = torch.vstack((labels_dataset[key], causal_labels[key]))

        for idx, img in enumerate(images):
            color_bar = causal_labels["color_bar"][idx]
            color_digit = causal_labels["color_digit"][idx]

            new_img = alter_digitbar_img(img, color_digit, color_bar)
            new_img = Image.fromarray(new_img, mode="RGB")
            img_dataset.append(new_img)

    # img_dataset = torch.vstack(
    #     [
    #         torchvision.transforms.functional.pil_to_tensor(x).reshape(1, 3, 28, 28)
    #         for x in img_dataset
    #     ]
    # )

    # print(img_dataset.shape)
    print(labels_dataset.keys())
    keys = ["digit", "color_digit", "color_bar", "distr_idx"]
    label_tensor = torch.zeros((len(img_dataset), len(keys)))
    # convert the labels from a list of dictionaries to a tensor array
    for idx, key in enumerate(keys):
        label_tensor[:, idx] = torch.Tensor(labels_dataset[key]).squeeze()
    print(label_tensor.shape)

    intervention_target_tensor = torch.zeros((len(img_dataset), 3), dtype=torch.int)
    intervention_target_tensor[:] = torch.Tensor(labels_dataset["intervention_targets"])
    print(intervention_target_tensor.shape)
    print(root)

    # save the actual data to disc now
    root = Path(root)
    imgs_fname = root / "CausalDigitBarMNIST" / "chainv2" / "chain-imgs-train.pt"
    labels_fname = root / "CausalDigitBarMNIST" / "chainv2" / "chain-labels-train.pt"
    targets_fname = root / "CausalDigitBarMNIST" / "chainv2" / "chain-targets-train.pt"
    imgs_fname.parent.mkdir(exist_ok=True, parents=True)

    torch.save(img_dataset, imgs_fname)
    torch.save(label_tensor, labels_fname)
    torch.save(intervention_target_tensor, targets_fname)
