"""Collision-avoidance constructions used in the Task P1 comparison."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm


def _normalized(direction: np.ndarray) -> np.ndarray:
    """Convert a nonzero direction to a floating-point unit vector."""
    direction = np.asarray(direction, dtype=float)
    magnitude = np.linalg.norm(direction)
    if magnitude == 0.0:
        raise ValueError("direction must be nonzero")
    return direction / magnitude


def heuristic_margin(
    mu: np.ndarray,
    covariance: np.ndarray,
    collision_radius: float,
    alpha: float,
    direction: np.ndarray,
) -> dict[str, object]:
    """Place the foot using a largest-eigenvalue heuristic margin."""
    normalized_direction = _normalized(direction)
    lambda_max = np.linalg.eigvalsh(covariance)[-1]
    sigma_max = np.sqrt(lambda_max)
    required_distance = collision_radius + alpha * sigma_max
    foot_position = np.asarray(mu) + required_distance * normalized_direction
    return {
        "foot_position": foot_position,
        "required_distance": required_distance,
        "lambda_max": lambda_max,
        "sigma_max": sigma_max,
        "normalized_direction": normalized_direction,
    }


def confidence_ellipse(
    mu: np.ndarray,
    covariance: np.ndarray,
    collision_radius: float,
    epsilon: float,
    direction: np.ndarray,
) -> dict[str, object]:
    """Place the foot beyond the support of an inflated confidence ellipse."""
    normalized_direction = _normalized(direction)
    chi2_quantile = chi2.ppf(1.0 - epsilon, df=2)
    directional_variance = normalized_direction.T @ covariance @ normalized_direction
    ellipse_directional_radius = np.sqrt(chi2_quantile * directional_variance)
    required_distance = collision_radius + ellipse_directional_radius
    foot_position = np.asarray(mu) + required_distance * normalized_direction

    # This is a conservative confidence-region construction, not the exact
    # circular collision chance constraint.
    return {
        "foot_position": foot_position,
        "required_distance": required_distance,
        "ellipse_directional_radius": ellipse_directional_radius,
        "chi2_quantile": chi2_quantile,
        "normalized_direction": normalized_direction,
    }


def directional_CC(
    mu: np.ndarray,
    covariance: np.ndarray,
    collision_radius: float,
    epsilon: float,
    direction: np.ndarray,
) -> dict[str, object]:
    """Place the foot on a fixed-direction Gaussian chance boundary."""
    normalized_direction = _normalized(direction)
    beta = norm.ppf(1.0 - epsilon)
    direction_sigma = np.sqrt(
        normalized_direction.T @ covariance @ normalized_direction
    )
    required_distance = collision_radius + beta * direction_sigma
    foot_position = np.asarray(mu) + required_distance * normalized_direction

    # For fixed n, this is the exact deterministic reformulation of the
    # one-dimensional Gaussian half-space chance constraint. For the original
    # circular collision event, it is a conservative sufficient condition.
    return {
        "foot_position": foot_position,
        "required_distance": required_distance,
        "direction_sigma": direction_sigma,
        "beta": beta,
        "normalized_direction": normalized_direction,
    }
