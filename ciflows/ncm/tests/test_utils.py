import pytest
import torch
from ciflows.ncm import (
    log,
    expand_do,
    check_equal,
    soft_equals,
    cross_entropy_compare,
)  # Replace `your_module` with actual module name


def test_log():
    x = torch.tensor([1.0, 2.0, 3.0])
    result = log(x)
    expected = torch.log(x + 1e-8)
    assert torch.allclose(result, expected), "log function failed"


def test_expand_do_scalar():
    result = expand_do(5.0, 3)
    expected = torch.tensor([[5.0], [5.0], [5.0]])
    assert torch.allclose(result, expected), "expand_do failed for scalar input"


def test_expand_do_tensor():
    val = torch.tensor([1.0, 2.0])
    result = expand_do(val, 3)
    expected = val.expand(3, 2)
    assert torch.allclose(result, expected), "expand_do failed for tensor input"


def test_check_equal_scalar():
    input_tensor = torch.tensor([[3.0], [3.0], [5.0]])
    result = check_equal(input_tensor, 3.0)
    expected = torch.tensor([True, True, False])
    assert torch.equal(result, expected), "check_equal failed for scalar comparison"


def test_check_equal_tensor():
    input_tensor = torch.tensor([[1, 2], [1, 2], [3, 4]])
    val = torch.tensor([1, 2])
    result = check_equal(input_tensor, val)
    expected = torch.tensor([True, True, False])
    assert torch.equal(result, expected), "check_equal failed for tensor comparison"


def test_soft_equals_scalar():
    input_tensor = torch.tensor([[3.0], [5.0], [7.0]])
    result = soft_equals(input_tensor, 4.0)
    expected = torch.tensor([1.0, 1.0, 3.0])
    assert torch.allclose(result, expected), "soft_equals failed for scalar comparison"


def test_soft_equals_tensor():
    input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
    val = torch.tensor([2, 3])
    result = soft_equals(input_tensor, val)
    expected = torch.tensor([2, 2, 6])
    assert torch.allclose(result, expected), "soft_equals failed for tensor comparison"


def test_cross_entropy_compare_1():
    input_tensor = torch.tensor([0.1, 0.5, 0.9])
    result = cross_entropy_compare(input_tensor, 1)
    expected = -torch.sum(log(input_tensor))
    assert torch.allclose(result, expected), "cross_entropy_compare failed for val=1"


def test_cross_entropy_compare_0():
    input_tensor = torch.tensor([0.1, 0.5, 0.9])
    result = cross_entropy_compare(input_tensor, 0)
    expected = -torch.sum(log(1 - input_tensor))
    assert torch.allclose(result, expected), "cross_entropy_compare failed for val=0"


def test_cross_entropy_compare_invalid():
    with pytest.raises(
        ValueError, match="Comparison to 0.5 of type <class 'float'> is not allowed."
    ):
        cross_entropy_compare(torch.tensor([0.2, 0.8]), 0.5)


def test_cross_entropy_compare_tensor():
    with pytest.raises(NotImplementedError):
        cross_entropy_compare(torch.tensor([0.2, 0.8]), torch.tensor([1]))
