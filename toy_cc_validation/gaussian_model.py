"""Reusable Gaussian covariance models for the Task P1 experiments."""

import numpy as np


DEFAULT_MEAN = np.array([0.0, 0.0])


def isotropic_covariance(sigma: float = 0.2) -> np.ndarray:
    """Return a 2D isotropic covariance with standard deviation ``sigma``."""
    return np.diag([sigma**2, sigma**2])


def anisotropic_covariance(
    sigma_x: float = 0.4, sigma_y: float = 0.1
) -> np.ndarray:
    """Return a diagonal 2D covariance with distinct axis uncertainties."""
    return np.diag([sigma_x**2, sigma_y**2])


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2D rotation matrix for an angle in radians."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def rotated_covariance(covariance: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a covariance by an orthogonal coordinate rotation."""
    rotation = rotation_matrix(theta)
    return rotation @ covariance @ rotation.T
