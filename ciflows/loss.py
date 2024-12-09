from math import prod, sqrt

import torch
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum over all dimensions except the first.
    :param x: Input tensor. Shape: (batch_size, ...)
    :return: Sum over all dimensions except the first. Shape: (batch_size,)
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)


def sample_orthonormal_vectors(x: torch.Tensor, n_samples: int = 1000):
    """Sample random vectors with scaled orthonormal columns for Hutchinson method.

    Parameters
    ----------
    x : torch.Tensor of shape (batch_size, ...)
        The reference data.
    n_samples : int, optional
        The number of samples to draw for the Hutchinson method, by default 1000.

    Returns
    -------
    torch.Tensor of shape (batch_size, total_dim, n_samples)
        The random vectors.
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])

    # sample from standard normal Gaussian ~ (batch_size, total_dim, n_samples)
    v = torch.randn(batch_size, total_dim, n_samples, dtype=x.dtype)

    # QR decomposition to get orthonormal columns
    q = torch.linalg.qr(v).Q.reshape(batch_size, total_dim, n_samples)

    # scale by sqrt(total_dim) to ensure E[v v^T] = I
    return q * sqrt(total_dim)


def volume_change_surrogate_old_transformer(
    x: torch.Tensor, encoder, decoder, hutchinson_samples: int = 1000, eta_samples=None
):
    """Compute volume change in change-of-variables formula using surrogate method.

    The surrogate is given by:
    $$
    v^T f_\\theta'(x) \\texttt{SG}(g_\\phi'(z) v).
    $$
    The gradient of the surrogate is the gradient of the volume change term.

    Parameters
    ----------
    x : torch.Tensor of shape (batch_size, ...)
        The input data.
    encoder : Callable
        The encoder fucntion, taking input x and returning v of shape (batch_size, latent_dim).
    decoder : Callable
        The decoder function, taking input z and returning xhat of shape (batch_size, ...).
    hutchinson_samples : int, optional
        The number of hutchinson samples to draw, by default 1000.

    Returns
    -------
    surrogate_loss : torch.Tensor
        The surrogate loss sum over all batches and hutchinson samples.
    v : torch.Tensor of shape (batch_size, latent_dim)
        The latent representation.
    xhat : torch.Tensor of shape (batch_size, ...)
        The reconstructed tensor of shape ``x``.
    """
    surrogate_loss = torch.Tensor([0.0]).to(x.device)

    # ensure gradients wrt x are computed
    with torch.set_grad_enabled(True):
        x.requires_grad_()

        # encode the data to get the latent representation
        v = encoder(x)
        B, n_patches, embed_dim = v.shape

        # project to the manifold and store projection distance for the regularization term
        # eta_samples = sample_orthonormal_vectors(x, hutchinson_samples)
        if eta_samples is None:
            # eta_samples = torch.zeros((B, n_patches, embed_dim, hutchinson_samples), device=x.device)

            # from transformer encoding
            eta_samples = sample_orthonormal_vectors(v, hutchinson_samples).to(x.device)
            eta_samples = eta_samples.reshape(B, n_patches, embed_dim, hutchinson_samples)

        for k in range(hutchinson_samples):
            eta = eta_samples[..., k]

            # compute forward-mode AD for the decoder
            # Note: this is efficient compared to reverse-mode AD because
            # it is assumed the decoder maps from a low-dimensional space
            # to a high-dimensional space
            with dual_level():
                # pass in (f(x), eta) to compute eta^T * f'(x)
                dual_v = make_dual(v, eta)

                # map the latent representation, and the Hutchinson samples
                # to the decoder high-dimensional manifold
                dual_x1 = decoder(dual_v)

                # v1 = \eta^T f'(x) (VJP)
                xhat, v1 = unpack_dual(dual_x1)

            # compute g'(v) eta using reverse-mode AD given:
            # - v = f(x)
            # - x = original data
            # - eta = random Hutchinson vector
            # v2 = g'(v) eta (JVP)
            (v2,) = grad(v, x, eta, create_graph=True)

            # detach v1 to avoid computing the gradient of the surrogate loss
            # print("v1 and v2 don't match?", v2.shape, v1.detach().shape, hutchinson_samples)
            res = torch.multiply(v2, v1.detach()).reshape(B, -1)
            surrogate_loss += torch.sum(res) / hutchinson_samples

    return surrogate_loss, v, xhat


def volume_change_surrogate(
    x: torch.Tensor,
    encoder,
    decoder,
    hutchinson_samples: int = 1000,
    eta_samples=None,
):
    """Compute volume change in change-of-variables formula using surrogate method.

    The surrogate is given by:
    $$
    v^T f_\\theta'(x) \\texttt{SG}(g_\\phi'(z) v).
    $$
    The gradient of the surrogate is the gradient of the volume change term.

    Parameters
    ----------
    x : torch.Tensor of shape (batch_size, ...)
        The input data.
    encoder : Callable
        The encoder fucntion, taking input x and returning v of shape (batch_size, latent_dim).
    decoder : Callable
        The decoder function, taking input z and returning xhat of shape (batch_size, ...).
    hutchinson_samples : int, optional
        The number of hutchinson samples to draw, by default 1000.

    Returns
    -------
    surrogate_loss : torch.Tensor
        The surrogate loss sum over all batches and hutchinson samples.
    v : torch.Tensor of shape (batch_size, latent_dim)
        The latent representation.
    xhat : torch.Tensor of shape (batch_size, ...)
        The reconstructed tensor of shape ``x``.
    """
    surrogate_loss = 0.0

    # ensure gradients wrt x are computed
    with torch.set_grad_enabled(True):
        x.requires_grad_()

        # encode the data to get the latent representation
        v = encoder(x)
        B, embed_dim = v.shape

        # project to the manifold and store projection distance for the regularization term
        # eta_samples = sample_orthonormal_vectors(x, hutchinson_samples)
        if eta_samples is None:
            eta_samples = sample_orthonormal_vectors(v, hutchinson_samples).to(x.device)
            eta_samples = eta_samples.reshape(B, embed_dim, hutchinson_samples)

        for k in range(hutchinson_samples):
            eta = eta_samples[..., k]

            # compute forward-mode AD for the decoder
            # Note: this is efficient compared to reverse-mode AD because
            # it is assumed the decoder maps from a low-dimensional space
            # to a high-dimensional space
            with dual_level():
                # pass in (f(x), eta) to compute eta^T * f'(x)
                dual_v = make_dual(v, eta)

                # map the latent representation, and the Hutchinson samples
                # to the decoder high-dimensional manifold
                dual_x1 = decoder(dual_v)

                # v1 = \eta^T f'(x) (VJP)
                xhat, v1 = unpack_dual(dual_x1)

            # compute g'(v) eta using reverse-mode AD given:
            # - v = f(x)
            # - x = original data
            # - eta = random Hutchinson vector
            # v2 = g'(v) eta (JVP)
            (v2,) = grad(v, x, eta, create_graph=True)

            # detach v1 to avoid computing the gradient of the surrogate loss
            # print("v1 and v2 don't match?", v2.shape, v1.detach().shape, hutchinson_samples)
            # print(v2.shape, v1.detach().shape)
            res = torch.multiply(v2, v1.detach()).reshape(B, -1)
            surrogate_loss += torch.sum(res, dim=1) / hutchinson_samples

    return surrogate_loss, v, xhat


def volume_change_surrogate_transformer(
    x: torch.Tensor,
    encoder,
    decoder,
    hutchinson_samples: int = 1000,
    eta_samples=None,
    transformer=True,
):
    """Compute volume change in change-of-variables formula using surrogate method.

    The surrogate is given by:
    $$
    v^T f_\\theta'(x) \\texttt{SG}(g_\\phi'(z) v).
    $$
    The gradient of the surrogate is the gradient of the volume change term.

    Parameters
    ----------
    x : torch.Tensor of shape (batch_size, ...)
        The input data.
    encoder : Callable
        The encoder fucntion, taking input x and returning v of shape (batch_size, latent_dim).
    decoder : Callable
        The decoder function, taking input z and returning xhat of shape (batch_size, ...).
    hutchinson_samples : int, optional
        The number of hutchinson samples to draw, by default 1000.

    Returns
    -------
    surrogate_loss : torch.Tensor
        The surrogate loss sum over all batches and hutchinson samples.
    v : torch.Tensor of shape (batch_size, latent_dim)
        The latent representation.
    xhat : torch.Tensor of shape (batch_size, ...)
        The reconstructed tensor of shape ``x``.
    """
    surrogate_loss = torch.Tensor([0.0]).to(x.device)

    # ensure gradients wrt x are computed
    with torch.set_grad_enabled(True):
        x.requires_grad_()

        # encode the data to get the latent representation
        v = encoder(x)
        B, n_patches, embed_dim = v.shape

        # ensure that patches are used as part of the batch dimension
        v = v.reshape(B, n_patches, -1)

        # project to the manifold and store projection distance for the regularization term
        # eta_samples = sample_orthonormal_vectors(x, hutchinson_samples)
        if eta_samples is None:
            # eta_samples = torch.zeros((B, n_patches, embed_dim, hutchinson_samples), device=x.device)

            # from transformer encoding
            eta_samples = sample_orthonormal_vectors(v, hutchinson_samples).to(x.device)
            eta_samples = eta_samples.reshape(B, n_patches, embed_dim, hutchinson_samples)

        for k in range(hutchinson_samples):
            eta = eta_samples[..., k]

            # compute forward-mode AD for the decoder
            # Note: this is efficient compared to reverse-mode AD because
            # it is assumed the decoder maps from a low-dimensional space
            # to a high-dimensional space
            with dual_level():
                # pass in (f(x), eta) to compute eta^T * f'(x)
                dual_v = make_dual(v, eta)

                # map the latent representation, and the Hutchinson samples
                # to the decoder high-dimensional manifold
                dual_x1 = decoder(dual_v)

                # v1 = \eta^T f'(x) (VJP)
                xhat, v1 = unpack_dual(dual_x1)

            # compute g'(v) eta using reverse-mode AD given:
            # - v = f(x)
            # - x = original data
            # - eta = random Hutchinson vector
            # v2 = g'(v) eta (JVP)
            (v2,) = grad(v, x, eta, create_graph=True)

            if transformer:
                # if patches are used as part of the batch dimension, only get
                # the last patch prediction of v1
                v1 = v1[:, -1, ...]

            # detach v1 to avoid computing the gradient of the surrogate loss
            # print("v1 and v2 don't match?", v2.shape, v1.detach().shape, hutchinson_samples)
            # print(v2.shape, v1.detach().shape)
            res = torch.multiply(v2, v1.detach()).reshape(B, -1)
            surrogate_loss += torch.sum(res) / hutchinson_samples

    return surrogate_loss, v, xhat
