import pytest
import torch
import numpy as np
from ciflows.ncm import NeuralDistribution, UniformDistribution, StandardNormalDistribution, MLP


def test_uniform_distribution_sample():
    """Test uniform distribution sampling returns correctly shaped tensors."""
    dist = UniformDistribution(["x", "y"], {"x": 2, "y": 3}, seed=42)
    samples = dist.sample(n=5, device="cpu")

    assert isinstance(samples, dict)
    assert set(samples.keys()) == {"x", "y"}
    assert samples["x"].shape == (5, 2)
    assert samples["y"].shape == (5, 3)
    assert (samples["x"] >= 0).all() and (samples["x"] <= 1).all()  # Uniform [0,1] check


def test_standard_normal_distribution_sample():
    """Test standard normal distribution sampling returns correctly shaped tensors."""
    dist = StandardNormalDistribution(["a", "b"], {"a": 1, "b": 2}, seed=123)
    samples = dist.sample(n=4, device="cpu")

    assert isinstance(samples, dict)
    assert set(samples.keys()) == {"a", "b"}
    assert samples["a"].shape == (4, 1)
    assert samples["b"].shape == (4, 2)
    assert torch.is_floating_point(samples["a"])  # Ensure outputs are floats


def test_random_state_reproducibility():
    """Ensure reproducibility when using a fixed random seed."""
    dist1 = UniformDistribution(["z"], {"z": 2}, seed=100)
    dist2 = UniformDistribution(["z"], {"z": 2}, seed=100)

    samples1 = dist1.sample(n=3, device="cpu")["z"]
    samples2 = dist2.sample(n=3, device="cpu")["z"]

    assert torch.allclose(samples1, samples2), "Random state is not reproducible!"


def test_device_assignment():
    """Ensure samples are correctly assigned to the specified device."""
    if torch.cuda.is_available():
        device = "cuda"
        dist = StandardNormalDistribution(["c"], {"c": 4}, seed=50)
        samples = dist.sample(n=2, device=device)
        assert samples["c"].device.type == "cuda"
    else:
        pytest.skip("CUDA not available")


def test_sample_size_mismatch():
    """Ensure missing variable sizes default to 1."""
    dist = UniformDistribution(["u", "v"], {}, seed=42)
    samples = dist.sample(n=3, device="cpu")

    assert samples["u"].shape == (3, 1)
    assert samples["v"].shape == (3, 1)


def test_neural_distribution_sample():
    """Test sampling from NeuralDistribution with valid neural network initialization."""
    hyperparams = {"h-layers": 3, "h-size": 64, "layer-norm": True}
    dist = NeuralDistribution(
        ["x", "y"], {"x": 2, "y": 3}, hyperparams, default_module=MLP, seed=42
    )

    samples = dist.sample(n=5, device="cpu")

    assert isinstance(samples, dict)
    assert set(samples.keys()) == {"x", "y"}
    assert samples["x"].shape == (5, 2)
    assert samples["y"].shape == (5, 3)
    assert torch.is_floating_point(samples["x"])


@pytest.mark.skip(reason="XXX fix")
def test_neural_distribution_random_state_reproducibility():
    """Ensure reproducibility with a fixed random seed."""
    hyperparams = {"h-layers": 2, "h-size": 128, "layer-norm": False}
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    dist1 = NeuralDistribution(["z"], {"z": 2}, hyperparams, default_module=MLP, seed=123)
    dist2 = NeuralDistribution(["z"], {"z": 2}, hyperparams, default_module=MLP, seed=123)

    samples1 = dist1.sample(n=3, device="cpu")["z"]
    samples2 = dist2.sample(n=3, device="cpu")["z"]
    assert torch.allclose(samples1, samples2), "Random state is not reproducible!"


def test_neural_distribution_device_assignment():
    """Ensure samples are assigned to the specified device."""
    hyperparams = {"h-layers": 2, "h-size": 128, "layer-norm": False}

    if torch.cuda.is_available():
        device = "cuda"
        dist = NeuralDistribution(["c"], {"c": 4}, hyperparams, default_module=MLP, seed=50)
        samples = dist.sample(n=2, device=device)
        assert samples["c"].device.type == "cuda"
    else:
        pytest.skip("CUDA not available")
