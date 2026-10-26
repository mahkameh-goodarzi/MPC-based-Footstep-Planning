"""Monte Carlo utilities for Gaussian collision validation."""

import numpy as np


def sample_gaussian(
    mu: np.ndarray,
    covariance: np.ndarray,
    number_of_samples: int,
    seed: int,
) -> np.ndarray:
    """Draw a reproducible set of Gaussian position samples."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, covariance, size=number_of_samples)


def collision_probability(
    samples: np.ndarray, foot: np.ndarray, collision_radius: float
) -> float:
    """Return the empirical probability of true circular collision."""
    collisions = np.linalg.norm(foot - samples, axis=1) <= collision_radius
    return float(np.mean(collisions))


def halfspace_probability(
    samples: np.ndarray,
    foot: np.ndarray,
    collision_radius: float,
    direction: np.ndarray,
) -> float:
    """Return the empirical probability of directional half-space violation."""
    normalized_direction = direction / np.linalg.norm(direction)
    violations = (foot - samples) @ normalized_direction < collision_radius
    return float(np.mean(violations))
