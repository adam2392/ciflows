import torch


def log(x):
    """
    Computes the natural logarithm of `x`, adding a small epsilon for numerical stability.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Logarithm of `x` with epsilon added.
    """
    return torch.log(x + 1e-8)


def expand_do(val, n):
    """
    Expands a scalar or tensor to a tensor of shape (n, 1).

    Parameters
    ----------
    val : float or torch.Tensor
        Scalar value or tensor to be expanded.
    n : int
        Number of times to expand.

    Returns
    -------
    torch.Tensor
        Expanded tensor of shape (n, 1).
    """
    if torch.is_tensor(val):
        return val.expand(n, *val.shape)
    else:
        return torch.full((n, 1), val, dtype=torch.float)


def check_equal(input, val):
    """
    Checks element-wise equality between `input` and `val`.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    val : torch.Tensor or scalar
        Value to compare against.

    Returns
    -------
    torch.Tensor
        Boolean tensor indicating equality along the first dimension.
    """
    if torch.is_tensor(val):
        return torch.all(input == val.expand_as(input), dim=1)
    else:
        return input.eq(val).squeeze()


def soft_equals(input, val):
    """
    Computes the sum of absolute differences between `input` and `val`.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    val : torch.Tensor or scalar
        Value to compare against.

    Returns
    -------
    torch.Tensor
        Sum of absolute differences along the first dimension.
    """
    if torch.is_tensor(val):
        return torch.sum(torch.abs(input - val.expand_as(input)), dim=1)
    else:
        return torch.abs(input - val).squeeze()


def cross_entropy_compare(input, val):
    """
    Computes a simple cross-entropy loss-like function for binary values.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor with values in (0,1).
    val : {0, 1}
        Target value for comparison.

    Returns
    -------
    torch.Tensor
        Summed cross-entropy loss.

    Raises
    ------
    ValueError
        If `val` is not 0 or 1.
    NotImplementedError
        If `val` is a tensor.
    """
    if torch.is_tensor(val):
        raise NotImplementedError("Tensor comparison is not implemented.")
    if val == 1:
        return -log(input).sum()
    elif val == 0:
        return -log(1 - input).sum()
    else:
        raise ValueError(f"Comparison to {val} of type {type(val)} is not allowed.")
