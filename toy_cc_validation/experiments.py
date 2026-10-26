"""Run the complete standalone Task P1 chance-constraint validation suite."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from collision_methods import confidence_ellipse, directional_CC, heuristic_margin
from gaussian_model import (
    DEFAULT_MEAN,
    anisotropic_covariance,
    isotropic_covariance,
    rotation_matrix,
    rotated_covariance,
)
from monte_carlo import (
    collision_probability,
    halfspace_probability,
    sample_gaussian,
)
from visualization import generate_all_figures


COLLISION_RADIUS = 0.35
NUMBER_OF_SAMPLES = 200_000
RANDOM_SEED = 42
MONTE_CARLO_TOLERANCE = 0.005
SAFETY_TOLERANCE = 0.002
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
COMBINED_COLUMNS = [
    "experiment",
    "method",
    "epsilon",
    "angle_deg",
    "rotation_angle_deg",
    "covariance_scale",
    "distance_required",
    "direction_sigma",
    "beta",
    "probability_halfspace",
    "probability_collision",
    "number_of_samples",
    "extra_margin",
]


def experiment_1_risk_level(
    mu: np.ndarray, covariance: np.ndarray
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Sweep allowed risk while holding the anisotropic model fixed."""
    epsilon_values = [0.20, 0.10, 0.05, 0.01]
    direction = np.array([1.0, 0.0])
    samples = sample_gaussian(mu, covariance, NUMBER_OF_SAMPLES, RANDOM_SEED)
    records = []

    for epsilon in epsilon_values:
        result = directional_CC(
            mu, covariance, COLLISION_RADIUS, epsilon, direction
        )
        records.append(
            {
                "experiment": "risk_level",
                "method": "Directional CC",
                "epsilon": epsilon,
                "angle_deg": np.nan,
                "rotation_angle_deg": np.nan,
                "covariance_scale": np.nan,
                "distance_required": result["required_distance"],
                "direction_sigma": result["direction_sigma"],
                "beta": result["beta"],
                "probability_halfspace": halfspace_probability(
                    samples,
                    result["foot_position"],
                    COLLISION_RADIUS,
                    result["normalized_direction"],
                ),
                "probability_collision": collision_probability(
                    samples, result["foot_position"], COLLISION_RADIUS
                ),
                "number_of_samples": NUMBER_OF_SAMPLES,
                "extra_margin": result["required_distance"] - COLLISION_RADIUS,
            }
        )

    data = pd.DataFrame(records)
    checks = {
        "halfspace risk follows epsilon": bool(
            np.all(
                np.abs(data["probability_halfspace"] - data["epsilon"])
                < MONTE_CARLO_TOLERANCE
            )
        ),
        "collision probability <= epsilon": bool(
            np.all(
                data["probability_collision"]
                <= data["epsilon"] + SAFETY_TOLERANCE
            )
        ),
        "distance increases as epsilon decreases": bool(
            np.all(np.diff(data["distance_required"]) > 0.0)
        ),
    }
    return data, checks


def experiment_2_method_comparison(
    mu: np.ndarray, covariance: np.ndarray
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, dict[str, object]]]:
    """Compare all three Task P1 methods on identical Gaussian samples."""
    epsilon = 0.05
    alpha = 2.0
    direction = np.array([1.0, 0.0])
    samples = sample_gaussian(mu, covariance, NUMBER_OF_SAMPLES, RANDOM_SEED)
    method_results = {
        "Heuristic": heuristic_margin(
            mu, covariance, COLLISION_RADIUS, alpha, direction
        ),
        "Confidence Ellipse": confidence_ellipse(
            mu, covariance, COLLISION_RADIUS, epsilon, direction
        ),
        "Directional CC": directional_CC(
            mu, covariance, COLLISION_RADIUS, epsilon, direction
        ),
    }
    records = []

    for method, result in method_results.items():
        is_directional = method == "Directional CC"
        direction_sigma = (
            result["direction_sigma"]
            if is_directional
            else (
                np.sqrt(direction @ covariance @ direction)
                if method == "Confidence Ellipse"
                else np.nan
            )
        )
        records.append(
            {
                "experiment": "method_comparison",
                "method": method,
                "epsilon": epsilon,
                "angle_deg": np.nan,
                "rotation_angle_deg": np.nan,
                "covariance_scale": np.nan,
                "distance_required": result["required_distance"],
                "direction_sigma": direction_sigma,
                "beta": result["beta"] if is_directional else np.nan,
                "probability_halfspace": (
                    halfspace_probability(
                        samples,
                        result["foot_position"],
                        COLLISION_RADIUS,
                        result["normalized_direction"],
                    )
                    if is_directional
                    else np.nan
                ),
                "probability_collision": collision_probability(
                    samples, result["foot_position"], COLLISION_RADIUS
                ),
                "number_of_samples": NUMBER_OF_SAMPLES,
                "extra_margin": result["required_distance"] - COLLISION_RADIUS,
            }
        )

    data = pd.DataFrame(records)
    indexed = data.set_index("method")
    checks = {
        "Directional CC is safe": bool(
            indexed.loc["Directional CC", "probability_collision"]
            <= epsilon + SAFETY_TOLERANCE
        ),
        "Directional CC distance < Confidence Ellipse distance": bool(
            indexed.loc["Directional CC", "distance_required"]
            < indexed.loc["Confidence Ellipse", "distance_required"]
        ),
    }
    return data, checks, method_results


def experiment_3_direction_dependence(
    mu: np.ndarray, covariance: np.ndarray
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Sweep the fixed separating direction through a complete circle."""
    epsilon = 0.05
    samples = sample_gaussian(mu, covariance, NUMBER_OF_SAMPLES, RANDOM_SEED)
    records = []

    for angle_deg in range(0, 361, 10):
        theta = np.radians(angle_deg)
        direction = np.array([np.cos(theta), np.sin(theta)])
        result = directional_CC(
            mu, covariance, COLLISION_RADIUS, epsilon, direction
        )
        records.append(
            {
                "experiment": "direction_dependence",
                "method": "Directional CC",
                "epsilon": epsilon,
                "angle_deg": float(angle_deg),
                "rotation_angle_deg": np.nan,
                "covariance_scale": np.nan,
                "distance_required": result["required_distance"],
                "direction_sigma": result["direction_sigma"],
                "beta": result["beta"],
                "probability_halfspace": halfspace_probability(
                    samples,
                    result["foot_position"],
                    COLLISION_RADIUS,
                    result["normalized_direction"],
                ),
                "probability_collision": collision_probability(
                    samples, result["foot_position"], COLLISION_RADIUS
                ),
                "number_of_samples": NUMBER_OF_SAMPLES,
                "extra_margin": result["required_distance"] - COLLISION_RADIUS,
            }
        )

    data = pd.DataFrame(records)
    maximum = data["distance_required"].max()
    minimum = data["distance_required"].min()
    maximum_angles = set(
        data.loc[np.isclose(data["distance_required"], maximum), "angle_deg"]
    )
    minimum_angles = set(
        data.loc[np.isclose(data["distance_required"], minimum), "angle_deg"]
    )
    checks = {
        "required distance varies with direction": bool(maximum > minimum),
        "extrema match anisotropic covariance axes": bool(
            maximum_angles == {0.0, 180.0, 360.0}
            and minimum_angles == {90.0, 270.0}
        ),
    }
    return data, checks


def experiment_4_rotation_invariance(
    mu: np.ndarray, covariance: np.ndarray
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Rotate samples, covariance, and direction by the same angles."""
    epsilon = 0.05
    base_direction = np.array([1.0, 0.0])
    base_samples = sample_gaussian(mu, covariance, NUMBER_OF_SAMPLES, RANDOM_SEED)
    records = []

    for angle_deg in [0.0, 45.0, 90.0]:
        theta = np.radians(angle_deg)
        rotation = rotation_matrix(theta)
        covariance_rotated = rotated_covariance(covariance, theta)
        direction_rotated = rotation @ base_direction
        samples_rotated = mu + (base_samples - mu) @ rotation.T
        result = directional_CC(
            mu,
            covariance_rotated,
            COLLISION_RADIUS,
            epsilon,
            direction_rotated,
        )
        records.append(
            {
                "experiment": "rotation_invariance",
                "method": "Directional CC",
                "epsilon": epsilon,
                "angle_deg": np.nan,
                "rotation_angle_deg": angle_deg,
                "covariance_scale": np.nan,
                "distance_required": result["required_distance"],
                "direction_sigma": result["direction_sigma"],
                "beta": result["beta"],
                "probability_halfspace": halfspace_probability(
                    samples_rotated,
                    result["foot_position"],
                    COLLISION_RADIUS,
                    result["normalized_direction"],
                ),
                "probability_collision": collision_probability(
                    samples_rotated, result["foot_position"], COLLISION_RADIUS
                ),
                "number_of_samples": NUMBER_OF_SAMPLES,
                "extra_margin": result["required_distance"] - COLLISION_RADIUS,
            }
        )

    data = pd.DataFrame(records)
    checks = {
        "directional sigma and distance are invariant": bool(
            np.ptp(data["direction_sigma"]) < 1e-10
            and np.ptp(data["distance_required"]) < 1e-10
        ),
        "empirical probabilities are invariant": bool(
            np.ptp(data["probability_halfspace"]) < 1e-12
            and np.ptp(data["probability_collision"]) < 1e-12
        ),
    }
    return data, checks


def experiment_5_deterministic_limit(
    mu: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Demonstrate convergence to deterministic separation as Sigma -> 0."""
    epsilon = 0.05
    direction = np.array([1.0, 0.0])
    scale_values = [1e-2, 1e-4, 1e-6, 1e-8]
    records = []

    for scale in scale_values:
        covariance = scale * np.eye(2)
        result = directional_CC(
            mu, covariance, COLLISION_RADIUS, epsilon, direction
        )
        records.append(
            {
                "experiment": "deterministic_limit",
                "method": "Directional CC",
                "epsilon": epsilon,
                "angle_deg": np.nan,
                "rotation_angle_deg": np.nan,
                "covariance_scale": scale,
                "distance_required": result["required_distance"],
                "direction_sigma": result["direction_sigma"],
                "beta": result["beta"],
                "probability_halfspace": np.nan,
                "probability_collision": np.nan,
                "number_of_samples": np.nan,
                "extra_margin": result["required_distance"] - COLLISION_RADIUS,
                "distance_minus_R": result["required_distance"]
                - COLLISION_RADIUS,
            }
        )

    data = pd.DataFrame(records)
    differences = data["distance_minus_R"].to_numpy()
    checks = {
        "distance_required approaches R": bool(
            np.all(np.diff(differences) < 0.0) and differences[-1] < 1e-3
        )
    }
    return data, checks


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write a CSV with explicit NaN markers and quiet known cast warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
        data.to_csv(path, index=False, na_rep="NaN")


def _print_experiment_summary(
    experiment_number: int, checks: dict[str, bool]
) -> bool:
    """Print all checks for one experiment and return their conjunction."""
    passed = all(checks.values())
    print(f"Experiment {experiment_number}: {'PASS' if passed else 'FAIL'}")
    for description, condition in checks.items():
        print(f"  - {description}: {'PASS' if condition else 'FAIL'}")
    return passed


def main() -> None:
    """Execute experiments, save results, generate figures, and summarize."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    mu = DEFAULT_MEAN.copy()
    covariance = anisotropic_covariance(sigma_x=0.4, sigma_y=0.1)

    risk_data, risk_checks = experiment_1_risk_level(mu, covariance)
    method_data, method_checks, method_results = experiment_2_method_comparison(
        mu, covariance
    )
    direction_data, direction_checks = experiment_3_direction_dependence(
        mu, covariance
    )
    rotation_data, rotation_checks = experiment_4_rotation_invariance(
        mu, covariance
    )
    deterministic_data, deterministic_checks = experiment_5_deterministic_limit(mu)

    _write_csv(risk_data, RESULTS_DIRECTORY / "risk_sweep.csv")
    _write_csv(method_data, RESULTS_DIRECTORY / "method_comparison.csv")
    _write_csv(direction_data, RESULTS_DIRECTORY / "direction_sweep.csv")
    _write_csv(rotation_data, RESULTS_DIRECTORY / "rotation_test.csv")
    _write_csv(deterministic_data, RESULTS_DIRECTORY / "deterministic_limit.csv")

    combined_data = pd.concat(
        [risk_data, method_data, direction_data, rotation_data, deterministic_data],
        ignore_index=True,
    ).reindex(columns=COMBINED_COLUMNS)
    _write_csv(combined_data, RESULTS_DIRECTORY / "results.csv")

    generate_all_figures(
        RESULTS_DIRECTORY,
        mu,
        covariance,
        COLLISION_RADIUS,
        method_results,
        risk_data,
        method_data,
        direction_data,
        rotation_data,
        deterministic_data,
    )

    print("## Task P1 Automatic Validation Summary")
    print()
    experiment_statuses = [
        _print_experiment_summary(1, risk_checks),
        _print_experiment_summary(2, method_checks),
        _print_experiment_summary(3, direction_checks),
        _print_experiment_summary(4, rotation_checks),
        _print_experiment_summary(5, deterministic_checks),
    ]
    print()
    print(f"OVERALL TASK P1: {'PASS' if all(experiment_statuses) else 'FAIL'}")


if __name__ == "__main__":
    main()
