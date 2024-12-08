import numpy as np
import pytest

from ciflows.distributions.sampling import sample_well_conditioned_matrix


@pytest.mark.parametrize("d1, d2, cond_num", [(5, 3, 10), (10, 5, 5), (10, 20, 10)])
def test_well_conditioned_matrix(d1, d2, cond_num):
    tolerance = 1e-6

    # Generate the matrix
    A = sample_well_conditioned_matrix(d1, d2, cond_num)

    # Check dimensions
    assert A.shape == (d2, d1), f"Expected shape {(d2, d1)}, but got {A.shape}"

    # Check condition number with tolerance
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    computed_cond_num = S[0] / S[-1]  # largest singular value / smallest singular value
    assert (
        computed_cond_num <= cond_num + tolerance
    ), f"Condition number {computed_cond_num} exceeds the limit {cond_num} (with tolerance {tolerance})"


def sample_from_cov(cov_matrix, n_samples=1000):
    """
    Samples points from a multivariate normal distribution with a given covariance matrix.
    :param cov_matrix: Covariance matrix to sample from.
    :param n_samples: Number of samples to generate.
    :return: Samples of shape (n_samples, d).
    """
    mean = np.zeros(cov_matrix.shape[0])
    samples = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    return samples


def compute_correlation_matrix(samples):
    """
    Computes the correlation matrix from the samples.
    :param samples: A 2D array of shape (n_samples, d) where each column is a variable.
    :return: The correlation matrix.
    """
    return np.corrcoef(samples, rowvar=False)


# from scipy.linalg import logm, sqrtm
# import pytest


# def perturb_covariance_matrix(base_matrix, alpha):
#     """
#     Perturbs the base covariance matrix to create a new covariance matrix.
#     :param base_matrix: The base covariance matrix.
#     :param alpha: A parameter that controls the closeness of the new matrix to the base matrix.
#     :return: A perturbed covariance matrix.
#     """
#     d = base_matrix.shape[0]
#     perturbation = alpha * (np.random.rand(d, d) - 0.5)  # Random perturbation centered around 0
#     perturbation = (perturbation + perturbation.T) / 2  # Ensure it's symmetric
#     perturbed_matrix = base_matrix + perturbation

#     # Ensure positive definiteness
#     return (perturbed_matrix + perturbed_matrix.T) / 2 + np.eye(d) * 1e-5  # Small identity for stability

# def riemannian_distance(A, B):
#     """
#     Computes the Riemannian distance between two positive definite matrices A and B.
#     :param A: A positive definite matrix.
#     :param B: A positive definite matrix.
#     :return: The Riemannian distance.
#     """
#     A_inv_half = sqrtm(np.linalg.inv(A))
#     M = A_inv_half @ B @ A_inv_half
#     log_M = logm(M)
#     return np.linalg.norm(log_M, ord='fro')

# def test_covariance_sampling_and_distance():
#     d = 3  # Dimension of the covariance matrix
#     alpha = 0.1  # Closeness parameter

#     # Sample a base covariance matrix
#     cov_matrix_1 = sample_random_covariance_matrix(d)

#     # Generate a perturbed covariance matrix that is "closer"
#     cov_matrix_2 = perturb_covariance_matrix(cov_matrix_1, alpha)

#     # Generate a perturbed covariance matrix that is "closer"
#     cov_matrix_3 = perturb_covariance_matrix(cov_matrix_1, alpha * 10)

#     # Compute the Riemannian distance between the two matrices
#     distance = riemannian_distance(cov_matrix_1, cov_matrix_2)
#     distance2 = riemannian_distance(cov_matrix_1, cov_matrix_3)

#     # Print results for demonstration
#     print(f"Covariance Matrix 1:\n{cov_matrix_1}")
#     print(f"Covariance Matrix 2:\n{cov_matrix_2}")
#     print(f"Riemannian Distance: {distance:.4f}")

#     # Assert the distance is non-negative
#     assert distance >= 0, "Distance should be non-negative!"

#     assert distance2 < distance
