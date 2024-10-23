import numpy as np


def sample_well_conditioned_matrix(d1, d2, cond_num=10, seed=None):
    """Generate random well-conditioned matrix.

    Parameters
    ----------
    d1 : int
        Source dimension.
    d2 : int
        Target dimension.
    cond_num : int, optional
        Condition number, by default 10.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    A_new_normalized : np.ndarray of shape (d2, d1)
        Well-conditioned matrix.
    """
    rng = np.random.default_rng(seed)

    # Step 1: Random matrix A with Gaussian entries
    A = rng.standard_normal(size=(d2, d1))

    # Step 2: Perform SVD
    U, _, Vt = np.linalg.svd(A, full_matrices=False)

    # Step 3: Modify singular values to be well-conditioned
    sigma_min = 1
    sigma_max = cond_num
    S_new = np.linspace(sigma_max, sigma_min, min(d1, d2))

    # Step 4: Reconstruct the matrix with the new singular values
    A_new = np.dot(U * S_new, Vt)

    # Optional Step 5: Normalize by Frobenius norm (or another norm)
    A_new_normalized = A_new / np.linalg.norm(A_new, ord="fro")

    return A_new_normalized


def sample_random_covariance_matrix(d, seed=None):
    """
    Samples a random positive definite covariance matrix of dimension d.
    :param d: Dimension of the covariance matrix.
    :return: A d x d positive definite covariance matrix.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(d, d))
    return np.dot(A, A.T)  # A positive definite matrix


def sample_random_vector(d, min_val=None, max_val=None, seed=None):
    """Sample random d-dimensional vector.

    Parameters
    ----------
    d : _type_
        _description_
    min_val : _type_, optional
        _description_, by default None
    max_val : _type_, optional
        _description_, by default None
    seed : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    rng = np.random.default_rng(seed)
    if min_val is None:
        min_val = -1
    if max_val is None:
        max_val = 1
    return rng.uniform(min_val, max_val, size=(d,))

# Example usage
# d = 5  # Dimensionality
# k = 4  # Number of vectors
# vectors = sample_spherical_vectors(d, k)

# print("Sampled Vectors:")
# print(vectors)
# print("\nCosine Distances:")
# cosine_distances = 1 - np.dot(vectors, vectors.T)
# print(cosine_distances.round(3))

def regular_simplex_points(d, k):
    """Generate k equidistant points on the surface of a d-dimensional unit sphere."""
    assert k <= d + 1, "k must be less than or equal to d + 1"
    
    # Initialize a matrix to hold the points
    points = np.zeros((k, d))
    
    # Calculate the points
    for i in range(k):
        # Generate the points of the simplex
        for j in range(d):
            if j < k - 1:
                points[i, j] = np.random.normal(size=(1,))
            else:
                points[i, j] = -np.sum(points[i, :k-1])  # Ensure that they sum to 0
        points[i] /= np.linalg.norm(points[i])  # Normalize to unit length

    return points

# Example usage
d = 5  # Dimensionality
k = 4  # Number of vectors
vectors = regular_simplex_points(d, k)

print("Sampled Vectors:")
print(vectors)

# Calculating cosine distances
cosine_distances = 1 - np.dot(vectors, vectors.T)
print("\nCosine Distances:")
print(cosine_distances.round(3))
# def sample_cov_matrix(d, base_correlation=0.5):
#     """
#     Generates a d x d positive definite covariance matrix with specified base correlation between variables.
#     :param d: Dimension of the covariance matrix.
#     :param base_correlation: Correlation between variables, applied to all off-diagonal elements.
#     :return: A d x d positive definite covariance matrix.
#     """
#     assert -1 <= base_correlation <= 1, "Correlation must be between -1 and 1"

#     # Initialize the covariance matrix
#     cov_matrix = np.eye(
#         d
#     )  # Start with an identity matrix (unit variance for each variable)

#     # Set the off-diagonal elements to the base correlation
#     for i in range(d):
#         for j in range(i + 1, d):
#             cov_matrix[i, j] = cov_matrix[j, i] = base_correlation

#     # Make sure the matrix is positive definite (cov_matrix is symmetric and should remain PD)
#     return cov_matrix
