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
    q = torch.linalg.qr(v).Q.reshape(*x.shape, n_samples)

    # scale by sqrt(total_dim) to ensure E[v v^T] = I
    return q * sqrt(total_dim)


def volume_change_surrogate(
    x: torch.Tensor, encoder, decoder, hutchinson_samples: int = 1000
):
    """Compute volume change in change-of-variables formula using surrogate method.

    The surrogate is given by:
    $$
    v^T f_\theta'(x) \texttt{SG}(g_\phi'(z) v).
    $$
    The gradient of the surrogate is the gradient of the volume change term.

    Parameters
    ----------
    x : torch.Tensor of shape (batch_size, ...)
        The input data.
    encoder : Callable
        The encoder fucntion, taking input x and returning v of shape (batch_size, latent_dim).
    decoder : _type_
        The decoder function, taking input z and returning xhat of shape (batch_size, ...).
    hutchinson_samples : int, optional
        The number of hutchinson samples to draw, by default 1000.

    Returns
    -------
    torch.Tensor of shape (batch_size,)
        The surrogate loss, latent representation, and reconstructed tensor.
    """
    # project to the manifold and store projection distance fo rthe regularization term
    eta_samples = sample_orthonormal_vectors(x, hutchinson_samples)

    # ensure gradients wrt x are computed
    x.requires_grad_()

    # encode the data to get the latent representation
    v = encoder(x)

    surrogate_loss = torch.zeros_like(v[:, 0])

    for k in range(len(eta_samples)):
        eta = eta_samples[k]

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

        # detach v1 to avoid computing the gradient of the
        surrogate_loss += sum_except_batch(v2 * v1.detach()) / hutchinson_samples

    return surrogate_loss, v, xhat
