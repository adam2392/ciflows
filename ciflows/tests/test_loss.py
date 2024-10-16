from collections import namedtuple
from math import prod, sqrt

import torch
# Define both versions of your surrogate function
import torch.autograd as autograd

from ciflows.loss import (sample_orthonormal_vectors,
                          volume_change_surrogate_transformer)
from ciflows.vit import VisionTransformerDecoder, VisionTransformerEncoder

SurrogateOutput = namedtuple(
    "SurrogateOutput", ["surrogate", "z", "x1", "regularizations"]
)


def test_volume_change_surrogate_shape():
    """Test that volume_change_surrogate outputs the expected shapes."""

    # Define input and model configurations
    batch_size = 2
    img_size = 28
    patch_size = 14
    in_channels = 3
    embed_dim = 2352
    n_heads = 4
    hidden_dim = 1024
    n_layers = 3
    hutchinson_samples = 5  # fewer samples for the test

    # Instantiate the encoder and decoder
    encoder = VisionTransformerEncoder(
        img_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, n_layers
    )
    decoder = VisionTransformerDecoder(
        img_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, n_layers
    )

    # Generate a dummy input image batch
    x = torch.randn(batch_size, in_channels, img_size, img_size)

    # Run the function
    surrogate_loss, v, xhat = volume_change_surrogate_transformer(
        x, encoder, decoder, hutchinson_samples=hutchinson_samples
    )

    # Check that the output shapes are correct
    assert (
        surrogate_loss.ndim == 1
    ), f"Expected surrogate loss shape {(1,)}, but got {surrogate_loss.shape}"
    assert v.shape == (
        batch_size,
        (img_size // patch_size) ** 2,
        embed_dim,
    ), f"Expected latent representation shape {(batch_size, (img_size // patch_size)**2, embed_dim)}, but got {v.shape}"
    assert (
        xhat.shape == x.shape
    ), f"Expected reconstructed image shape {x.shape}, but got {xhat.shape}"


def sample_v(x: torch.Tensor, hutchinson_samples: int, manifold=None) -> torch.Tensor:
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data. Shape: (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :param manifold: Optional manifold on which the data lies. If provided,
        the vectors are sampled in the tangent space of the manifold.
    :return: Random vectors of shape (batch_size, ...)
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])

    if hutchinson_samples > total_dim:
        raise ValueError(
            f"Too many Hutchinson samples: got {hutchinson_samples}, \
                expected <= {total_dim}"
        )
    # M-FFF: More than one Hutchinson sample not implemented for M-FFF
    if manifold is not None and hutchinson_samples != 1:
        raise NotImplementedError(
            f"More than one Hutchinson sample not implemented for M-FFF, \
                {hutchinson_samples} requested."
        )

    if manifold is None:
        v = torch.randn(
            batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype
        )
        q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
        return q * sqrt(total_dim)
    # M-FFF: Sample v in the tangent space of the manifold at x
    # else:
    #     v = random_tangent_vec(manifold, x.detach(), n_samples=batch_size)
    #     v /= torch.norm(v, p=2, dim=list(range(1, len(v.shape))), keepdim=True)
    #     return v[..., None] * sqrt(total_dim)


def volume_change_surrogate_v1(
    x, encode, decode, hutchinson_samples=1, manifold=None, vs=None, transformer=True
):
    regularizations = {}
    surrogate = 0

    x.requires_grad_()
    z = encode(x)

    print(z.shape)
    if manifold is not None:
        z_projected = manifold.projection(z)
        regularizations["z_projection"] = torch.nn.functional.mse_loss(z, z_projected)
        z = z_projected

    if vs is None:
        vs = sample_v(z, hutchinson_samples, manifold)

    print("Inside volume change surrogate v1...")
    print(vs.shape)
    for k in range(hutchinson_samples):
        v = vs[..., k]
        with autograd.forward_ad.dual_level():
            dual_z = autograd.forward_ad.make_dual(z, v)
            dual_x1 = decode(dual_z)

            if manifold is not None:
                dual_x1_projected = manifold.projection(dual_x1)
                regularizations["x1_projection"] = torch.nn.functional.mse_loss(
                    autograd.forward_ad.unpack_dual(dual_x1_projected)[0],
                    autograd.forward_ad.unpack_dual(dual_x1)[0],
                )
                dual_x1 = dual_x1_projected

            x1, v1 = autograd.forward_ad.unpack_dual(dual_x1)

        (v2,) = autograd.grad(z, x, v, create_graph=True)

        if transformer:
            v1 = v1[:, -1, ...]

        surrogate += (v2 * v1.detach()).sum() / hutchinson_samples

    return SurrogateOutput(surrogate, z, x1, regularizations)


# Compare both functions
def test_compare_surrogates():
    # Define input and model configurations
    batch_size = 2
    img_size = 28
    patch_size = 14
    in_channels = 3
    embed_dim = 200
    n_heads = 4
    hidden_dim = 1024
    n_layers = 3
    hutchinson_samples = 5  # fewer samples for the test

    # Instantiate the encoder and decoder
    encoder = VisionTransformerEncoder(
        img_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, n_layers
    )
    decoder = VisionTransformerDecoder(
        img_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, n_layers
    )
    # Generate a dummy input image batch
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    v = encoder(x)
    B, n_patches, embed_dim = v.shape
    hutchinson_samples = 4
    # eta_samples = torch.zeros((B, n_patches, embed_dim, hutchinson_samples))
    # from transformer encoding
    eta_samples = sample_orthonormal_vectors(v, hutchinson_samples)
    eta_samples = eta_samples.reshape(B, n_patches, embed_dim, hutchinson_samples)
    # for idx in range(n_patches):
    #     eta_samples[:, idx, ...] = sample_orthonormal_vectors(v, hutchinson_samples)

    # Run version 2
    surrogate_loss_v2, v_v2, xhat_v2 = volume_change_surrogate_transformer(
        x,
        encoder,
        decoder,
        hutchinson_samples=hutchinson_samples,
        eta_samples=eta_samples,
    )

    # Run version 1
    surrogate_output_v1 = volume_change_surrogate_v1(
        x,
        encoder,
        decoder,
        hutchinson_samples=hutchinson_samples,
        manifold=None,
        vs=eta_samples,
    )

    assert torch.allclose(
        surrogate_output_v1.surrogate, surrogate_loss_v2
    ), "Surrogate losses do not match."

    assert torch.allclose(
        surrogate_output_v1.z, v_v2
    ), "Latent representations do not match."
    assert torch.allclose(
        surrogate_output_v1.x1, xhat_v2
    ), "Reconstructions do not match."

    # Compare outputs
    print()
    print("Results:")
    # print(surrogate_output_v1)
    print(f"Surrogate V1: {surrogate_output_v1.surrogate}")
    print(f"Surrogate Loss V2: {surrogate_loss_v2}")

    # Compare latent representations
    print(
        f"Latent Representation Difference (z): {torch.abs(surrogate_output_v1.z - v_v2).sum()}"
    )

    # Compare reconstructions
    print(
        f"Reconstruction Difference (x1): {torch.abs(surrogate_output_v1.x1 - xhat_v2).sum()}"
    )
