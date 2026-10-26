"""Figure generation for the standalone Task P1 validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.stats import chi2


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Apply common layout, save, and close a figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure1_geometry(
    output_path: Path,
    mu: np.ndarray,
    covariance: np.ndarray,
    collision_radius: float,
    method_results: dict[str, dict[str, object]],
) -> None:
    """Plot the Gaussian confidence ellipse, collision radius, and feet."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ellipse_scale = np.sqrt(chi2.ppf(0.95, df=2) * eigenvalues)
    angles = np.linspace(0.0, 2.0 * np.pi, 400)
    unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    ellipse = mu[:, None] + eigenvectors @ np.diag(ellipse_scale) @ unit_circle

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.plot(ellipse[0], ellipse[1], label="95% confidence ellipse", linewidth=2)
    axis.scatter(mu[0], mu[1], color="black", marker="x", s=70, label="Human mean")
    axis.add_patch(
        Circle(
            mu,
            collision_radius,
            fill=False,
            linestyle="--",
            color="gray",
            label="Collision radius R",
        )
    )
    markers = {"Heuristic": "s", "Confidence Ellipse": "^", "Directional CC": "o"}
    for method, result in method_results.items():
        foot = np.asarray(result["foot_position"])
        axis.scatter(foot[0], foot[1], marker=markers[method], s=70, label=method)
    axis.set_title("Task P1 collision-avoidance geometry")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    _save_figure(figure, output_path)


def figure2_risk_validation(output_path: Path, risk_data: pd.DataFrame) -> None:
    """Plot empirical directional and circular risks against epsilon."""
    data = risk_data.sort_values("epsilon")
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(data["epsilon"], data["probability_halfspace"], "o-", label="Half-space")
    axis.plot(data["epsilon"], data["probability_collision"], "s-", label="Circular collision")
    axis.plot(data["epsilon"], data["epsilon"], "k--", label="Reference y = x")
    axis.set_xlabel("Allowed risk epsilon")
    axis.set_ylabel("Empirical probability")
    axis.set_title("Risk-level validation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _save_figure(figure, output_path)


def figure3_safety_conservatism(
    output_path: Path, method_data: pd.DataFrame
) -> None:
    """Compare geometric separation and empirical risk on separate axes."""
    methods = method_data["method"].tolist()
    positions = np.arange(len(methods))
    figure, (distance_axis, risk_axis) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    distance_axis.bar(positions, method_data["distance_required"], color="tab:blue")
    distance_axis.set_xticks(positions, methods, rotation=15)
    distance_axis.set_ylabel("Required distance [m]")
    distance_axis.set_title("Geometric separation")
    distance_axis.grid(True, axis="y", alpha=0.3)
    risk_axis.bar(positions, method_data["probability_collision"], color="tab:orange")
    risk_axis.axhline(0.05, color="black", linestyle="--", label="epsilon = 0.05")
    risk_axis.set_xticks(positions, methods, rotation=15)
    risk_axis.set_ylabel("Empirical collision probability")
    risk_axis.set_title("Achieved collision risk")
    risk_axis.grid(True, axis="y", alpha=0.3)
    risk_axis.legend()
    figure.suptitle("Three-method safety and conservatism comparison")
    _save_figure(figure, output_path)


def figure4_direction_effect(
    output_path: Path, direction_data: pd.DataFrame
) -> None:
    """Plot the full-circle directional separation requirement."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(direction_data["angle_deg"], direction_data["distance_required"], "o-", markersize=3)
    axis.set_xlabel("Direction angle [deg]")
    axis.set_ylabel("Required distance [m]")
    axis.set_title("Effect of anisotropic uncertainty direction")
    axis.set_xlim(0.0, 360.0)
    axis.set_xticks(np.arange(0.0, 361.0, 45.0))
    axis.grid(True, alpha=0.3)
    _save_figure(figure, output_path)


def figure5_rotation(output_path: Path, rotation_data: pd.DataFrame) -> None:
    """Plot empirical probabilities under common coordinate rotations."""
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(rotation_data["rotation_angle_deg"], rotation_data["probability_halfspace"], "o-", label="Half-space")
    axis.plot(rotation_data["rotation_angle_deg"], rotation_data["probability_collision"], "s-", label="Circular collision")
    axis.set_xlabel("Common rotation angle [deg]")
    axis.set_ylabel("Empirical probability")
    axis.set_title("Rotation invariance")
    axis.set_xticks(rotation_data["rotation_angle_deg"])
    axis.grid(True, alpha=0.3)
    axis.legend()
    _save_figure(figure, output_path)


def figure6_deterministic_limit(
    output_path: Path, deterministic_data: pd.DataFrame
) -> None:
    """Plot convergence of the uncertainty margin as covariance vanishes."""
    data = deterministic_data.sort_values("covariance_scale")
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.loglog(data["covariance_scale"], data["distance_minus_R"], "o-")
    axis.set_xlabel("Isotropic covariance scale")
    axis.set_ylabel("Required distance - R [m]")
    axis.set_title("Deterministic limit")
    axis.grid(True, which="both", alpha=0.3)
    _save_figure(figure, output_path)


def generate_all_figures(
    results_directory: Path,
    mu: np.ndarray,
    covariance: np.ndarray,
    collision_radius: float,
    method_results: dict[str, dict[str, object]],
    risk_data: pd.DataFrame,
    method_data: pd.DataFrame,
    direction_data: pd.DataFrame,
    rotation_data: pd.DataFrame,
    deterministic_data: pd.DataFrame,
) -> None:
    """Generate all six Task P1 figures."""
    figure1_geometry(results_directory / "figure1_geometry.png", mu, covariance, collision_radius, method_results)
    figure2_risk_validation(results_directory / "figure2_risk_validation.png", risk_data)
    figure3_safety_conservatism(results_directory / "figure3_safety_conservatism.png", method_data)
    figure4_direction_effect(results_directory / "figure4_direction_effect.png", direction_data)
    figure5_rotation(results_directory / "figure5_rotation.png", rotation_data)
    figure6_deterministic_limit(results_directory / "figure6_deterministic_limit.png", deterministic_data)
