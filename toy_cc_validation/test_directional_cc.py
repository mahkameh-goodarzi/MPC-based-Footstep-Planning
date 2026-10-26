"""Monte Carlo validation of a directional Gaussian chance constraint."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm


def check_label(passed: bool) -> str:
    """Return a readable status label for an automatic check."""
    return "PASS" if passed else "FAIL"


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of a nonzero direction vector."""
    norm_direction = np.linalg.norm(direction)
    if norm_direction == 0.0:
        raise ValueError("Direction vector must be nonzero.")
    return direction / norm_direction


def evaluate_boundary_case(
    epsilon: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    collision_radius: float,
    direction: np.ndarray,
    human_samples: np.ndarray,
) -> dict[str, float]:
    """Evaluate one case at its directional constraint boundary."""
    beta = norm.ppf(1.0 - epsilon)
    sigma_n = np.sqrt(direction.T @ sigma @ direction)
    required_distance = collision_radius + beta * sigma_n
    foot = mu + required_distance * direction

    lhs = direction.T @ (foot - mu)
    rhs = collision_radius + beta * sigma_n
    residual = lhs - rhs

    foot_to_human = foot - human_samples
    directional_separation = foot_to_human @ direction
    probability_halfspace = np.mean(directional_separation < collision_radius)
    probability_collision = np.mean(
        np.linalg.norm(foot_to_human, axis=1) <= collision_radius
    )

    return {
        "epsilon": epsilon,
        "beta": beta,
        "sigma_n": sigma_n,
        "required_distance": required_distance,
        "halfspace_probability": probability_halfspace,
        "collision_probability": probability_collision,
        "residual": residual,
    }


def main() -> None:
    """Run risk-level and directional-uncertainty validation experiments."""
    mu = np.array([0.0, 0.0])
    sigma = np.array(
        [
            [0.04, 0.0],
            [0.0, 0.01],
        ]
    )
    epsilon_values = [0.20, 0.10, 0.05, 0.01]
    collision_radius = 0.35  # Toy-test value, not Surena's final radius.
    direction = normalize_direction(np.array([1.0, 0.0]))
    number_of_samples = 200_000
    random_seed = 42
    rng = np.random.default_rng(random_seed)

    # For fixed n, the deterministic formula is the exact reformulation of the
    # one-dimensional Gaussian half-space chance constraint. It is NOT the exact
    # reformulation of the original circular collision chance constraint. For
    # the circular collision event, the directional half-space formulation is a
    # conservative sufficient condition: every circular collision is contained
    # in the directional half-space violation event.

    # Smaller epsilon means lower allowed collision risk. Because
    # beta = Phi^{-1}(1 - epsilon), beta increases as epsilon decreases.
    # Therefore, R + beta * sigma_n must increase as epsilon decreases.
    results = []
    for epsilon in epsilon_values:
        human_samples = rng.multivariate_normal(mu, sigma, size=number_of_samples)
        results.append(
            evaluate_boundary_case(
                epsilon,
                mu,
                sigma,
                collision_radius,
                direction,
                human_samples,
            )
        )

    print("## Directional Chance Constraint Risk-Level Sweep")
    print()
    print(f"R                        = {collision_radius:.8f}")
    print(f"Monte Carlo samples      = {number_of_samples}")
    print(f"random seed              = {random_seed}")
    print()
    print(
        "epsilon   beta       sigma_n    req_dist    "
        "halfspace_p   collision_p   residual"
    )
    for result in results:
        print(
            f"{result['epsilon']:<9.2f} "
            f"{result['beta']:<10.6f} "
            f"{result['sigma_n']:<10.6f} "
            f"{result['required_distance']:<11.6f} "
            f"{result['halfspace_probability']:<13.6f} "
            f"{result['collision_probability']:<13.6f} "
            f"{result['residual']:.3e}"
        )

    print()
    print("Automatic checks")
    for result in results:
        epsilon = result["epsilon"]
        boundary_check = abs(result["residual"]) < 1e-10
        halfspace_risk_check = (
            abs(result["halfspace_probability"] - epsilon) < 0.005
        )
        circular_safety_check = (
            result["collision_probability"] <= epsilon + 0.002
        )
        print(f"epsilon = {epsilon:.2f}")
        print(f"  1. Boundary equation check:    {check_label(boundary_check)}")
        print(f"  2. Halfspace risk check:       {check_label(halfspace_risk_check)}")
        print(f"  3. Circular collision safety:  {check_label(circular_safety_check)}")

    required_distances = [result["required_distance"] for result in results]
    monotonicity_check = all(
        later > earlier
        for earlier, later in zip(required_distances, required_distances[1:])
    )
    print(
        "Required distance increases as epsilon decreases: "
        f"{check_label(monotonicity_check)}"
    )

    # The directional chance constraint does not use a single isotropic
    # uncertainty radius. Instead, it projects the covariance onto the fixed
    # separating direction: sigma_n = sqrt(n.T @ Sigma @ n). Therefore, the
    # safety margin changes with direction when the covariance is anisotropic.
    direction_epsilon = 0.05
    direction_cases = [
        ("x-axis", np.array([1.0, 0.0])),
        ("y-axis", np.array([0.0, 1.0])),
        ("diagonal", np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])),
    ]
    direction_rng = np.random.default_rng(random_seed)
    direction_results = []
    for direction_name, direction_vector in direction_cases:
        unit_direction = normalize_direction(direction_vector)
        human_samples = direction_rng.multivariate_normal(
            mu, sigma, size=number_of_samples
        )
        result = evaluate_boundary_case(
            direction_epsilon,
            mu,
            sigma,
            collision_radius,
            unit_direction,
            human_samples,
        )
        direction_results.append((direction_name, unit_direction, result))

    print()
    print("## Directional Uncertainty Test")
    print()
    print(f"epsilon                  = {direction_epsilon:.8f}")
    print(f"R                        = {collision_radius:.8f}")
    print(f"Monte Carlo samples      = {number_of_samples}")
    print(f"random seed              = {random_seed}")
    print()
    print(
        "direction   nx       ny       sigma_n    req_dist    "
        "halfspace_p   collision_p   residual"
    )
    for direction_name, unit_direction, result in direction_results:
        print(
            f"{direction_name:<11} "
            f"{unit_direction[0]:<8.4f} "
            f"{unit_direction[1]:<8.4f} "
            f"{result['sigma_n']:<10.6f} "
            f"{result['required_distance']:<11.6f} "
            f"{result['halfspace_probability']:<13.6f} "
            f"{result['collision_probability']:<13.6f} "
            f"{result['residual']:.3e}"
        )

    print()
    print("Automatic checks")
    for direction_name, _, result in direction_results:
        boundary_check = abs(result["residual"]) < 1e-10
        halfspace_risk_check = (
            abs(result["halfspace_probability"] - direction_epsilon) < 0.005
        )
        circular_safety_check = (
            result["collision_probability"] <= direction_epsilon + 0.002
        )
        print(f"direction = {direction_name}")
        print(f"  1. Boundary equation check:    {check_label(boundary_check)}")
        print(f"  2. Halfspace risk check:       {check_label(halfspace_risk_check)}")
        print(f"  3. Circular collision safety:  {check_label(circular_safety_check)}")

    results_by_direction = {
        direction_name: result
        for direction_name, _, result in direction_results
    }
    sigma_x_check = abs(results_by_direction["x-axis"]["sigma_n"] - 0.20) < 1e-10
    sigma_y_check = abs(results_by_direction["y-axis"]["sigma_n"] - 0.10) < 1e-10
    sigma_diag_check = (
        abs(results_by_direction["diagonal"]["sigma_n"] - np.sqrt(0.025))
        < 1e-10
    )
    distance_ordering_check = (
        results_by_direction["x-axis"]["required_distance"]
        > results_by_direction["diagonal"]["required_distance"]
        > results_by_direction["y-axis"]["required_distance"]
    )
    print(f"sigma_x equals 0.20:             {check_label(sigma_x_check)}")
    print(f"sigma_y equals 0.10:             {check_label(sigma_y_check)}")
    print(
        "sigma_diag equals sqrt(0.025): "
        f"{check_label(sigma_diag_check)}"
    )
    print(
        "required_distance_x > required_distance_diag > required_distance_y: "
        f"{check_label(distance_ordering_check)}"
    )

    # The scalar n.T @ Sigma @ n is invariant under a common orthogonal
    # rotation, where n_rot = Q @ n and Sigma_rot = Q @ Sigma @ Q.T. Thus,
    # n_rot.T @ Sigma_rot @ n_rot = n.T @ Sigma @ n. This verifies that
    # directional uncertainty is a geometric quantity and does not depend on
    # the arbitrary orientation of the coordinate frame.
    rotation_epsilon = 0.05
    rotation_angle_deg = 45.0
    theta = np.radians(rotation_angle_deg)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    base_direction = normalize_direction(np.array([1.0, 0.0]))
    rotated_sigma = rotation_matrix @ sigma @ rotation_matrix.T
    rotated_direction = normalize_direction(rotation_matrix @ base_direction)

    rotation_beta = norm.ppf(1.0 - rotation_epsilon)
    sigma_n_base = np.sqrt(base_direction.T @ sigma @ base_direction)
    sigma_n_rotated = np.sqrt(
        rotated_direction.T @ rotated_sigma @ rotated_direction
    )
    required_distance_base = collision_radius + rotation_beta * sigma_n_base
    required_distance_rotated = (
        collision_radius + rotation_beta * sigma_n_rotated
    )

    rotation_rng = np.random.default_rng(random_seed)
    rotated_human_samples = rotation_rng.multivariate_normal(
        mu, rotated_sigma, size=number_of_samples
    )
    rotation_result = evaluate_boundary_case(
        rotation_epsilon,
        mu,
        rotated_sigma,
        collision_radius,
        rotated_direction,
        rotated_human_samples,
    )

    sigma_invariance_check = abs(sigma_n_base - sigma_n_rotated) < 1e-10
    distance_invariance_check = (
        abs(required_distance_base - required_distance_rotated) < 1e-10
    )
    rotation_halfspace_check = (
        abs(rotation_result["halfspace_probability"] - rotation_epsilon) < 0.005
    )
    rotation_collision_check = (
        rotation_result["collision_probability"] <= rotation_epsilon + 0.002
    )
    rotated_direction_norm_check = (
        abs(np.linalg.norm(rotated_direction) - 1.0) < 1e-12
    )

    print()
    print("## Rotation Invariance Test")
    print()
    print(f"rotation angle            = {rotation_angle_deg:.1f} degrees")
    print(f"base n                    = {base_direction}")
    print(f"rotated n                 = {rotated_direction}")
    print()
    print(f"base Sigma                =\n{sigma}")
    print(f"rotated Sigma             =\n{rotated_sigma}")
    print()
    print(f"base sigma_n              = {sigma_n_base:.12f}")
    print(f"rotated sigma_n           = {sigma_n_rotated:.12f}")
    print()
    print(f"base required distance    = {required_distance_base:.12f}")
    print(f"rotated required distance = {required_distance_rotated:.12f}")
    print()
    print(
        "halfspace probability     = "
        f"{rotation_result['halfspace_probability']:.8f}"
    )
    print(
        "collision probability     = "
        f"{rotation_result['collision_probability']:.8f}"
    )
    print()
    print("Automatic checks")
    print(
        "1. Directional sigma invariant:   "
        f"{check_label(sigma_invariance_check)}"
    )
    print(
        "2. Required distance invariant:   "
        f"{check_label(distance_invariance_check)}"
    )
    print(
        "3. Rotated halfspace risk:        "
        f"{check_label(rotation_halfspace_check)}"
    )
    print(
        "4. Rotated circular safety:       "
        f"{check_label(rotation_collision_check)}"
    )
    print(
        "5. Rotated direction is unit norm: "
        f"{check_label(rotated_direction_norm_check)}"
    )

    # When covariance vanishes, the probabilistic uncertainty term vanishes:
    # sqrt(n.T @ Sigma @ n) -> 0. Therefore, the directional chance constraint
    # smoothly reduces to the deterministic geometric separation
    # n.T @ (f - mu) >= R. This is an important consistency check.
    deterministic_epsilon = 0.05
    sigma_small = np.array(
        [
            [1e-12, 0.0],
            [0.0, 1e-12],
        ]
    )
    deterministic_direction = normalize_direction(np.array([1.0, 0.0]))
    deterministic_beta = norm.ppf(1.0 - deterministic_epsilon)
    deterministic_sigma_n = np.sqrt(
        deterministic_direction.T @ sigma_small @ deterministic_direction
    )
    uncertainty_margin = deterministic_beta * deterministic_sigma_n
    deterministic_required_distance = collision_radius + uncertainty_margin
    difference_from_radius = deterministic_required_distance - collision_radius

    small_sigma_check = deterministic_sigma_n < 1e-5
    small_margin_check = uncertainty_margin < 1e-5
    deterministic_distance_check = (
        abs(deterministic_required_distance - collision_radius) < 1e-5
    )
    not_inside_radius_check = deterministic_required_distance >= collision_radius

    print()
    print("## Deterministic Limit Test")
    print()
    print(f"epsilon                  = {deterministic_epsilon:.8f}")
    print(f"beta                     = {deterministic_beta:.12f}")
    print(f"R                        = {collision_radius:.12f}")
    print(f"sigma_n                  = {deterministic_sigma_n:.12e}")
    print(f"uncertainty margin       = {uncertainty_margin:.12e}")
    print(f"required distance        = {deterministic_required_distance:.12f}")
    print(f"difference from R        = {difference_from_radius:.12e}")
    print()
    print("Automatic checks")
    print(f"1. sigma_n < 1e-5:                       {check_label(small_sigma_check)}")
    print(f"2. uncertainty margin < 1e-5:            {check_label(small_margin_check)}")
    print(
        "3. abs(required distance - R) < 1e-5: "
        f"{check_label(deterministic_distance_check)}"
    )
    print(
        "4. required distance >= R:              "
        f"{check_label(not_inside_radius_check)}"
    )

    comparison_epsilon = 0.05
    comparison_direction = normalize_direction(np.array([1.0, 0.0]))
    alpha = 2.0

    # alpha = 2.0 is a heuristic tuning value from the Task P1 example. The
    # heuristic safety-margin method does not directly correspond to a
    # specified collision probability.
    lambda_max = np.linalg.eigvalsh(sigma)[-1]
    sigma_max = np.sqrt(lambda_max)
    safe_distance_heuristic = collision_radius + alpha * sigma_max

    # This is a conservative separation based on the supporting boundary of
    # the 2D confidence ellipse inflated by R. It is not the exact
    # reformulation of the circular collision chance constraint.
    chi2_quantile = chi2.ppf(1.0 - comparison_epsilon, df=2)
    ellipse_directional_radius = np.sqrt(
        chi2_quantile * (comparison_direction.T @ sigma @ comparison_direction)
    )
    safe_distance_ellipse = collision_radius + ellipse_directional_radius

    comparison_beta = norm.ppf(1.0 - comparison_epsilon)
    comparison_sigma_n = np.sqrt(
        comparison_direction.T @ sigma @ comparison_direction
    )
    safe_distance_directional = (
        collision_radius + comparison_beta * comparison_sigma_n
    )

    comparison_rng = np.random.default_rng(random_seed)
    comparison_human_samples = comparison_rng.multivariate_normal(
        mu, sigma, size=number_of_samples
    )
    method_distances = [
        ("Heuristic Margin", safe_distance_heuristic),
        ("Confidence Ellipse", safe_distance_ellipse),
        ("Directional CC", safe_distance_directional),
    ]
    comparison_results = []
    for method_name, safe_distance in method_distances:
        foot = mu + safe_distance * comparison_direction
        collision_probability = np.mean(
            np.linalg.norm(foot - comparison_human_samples, axis=1)
            <= collision_radius
        )
        comparison_results.append(
            {
                "method": method_name,
                "safe_distance": safe_distance,
                "extra_margin": safe_distance - collision_radius,
                "collision_probability": collision_probability,
            }
        )

    comparison_by_method = {
        result["method"]: result for result in comparison_results
    }
    all_distances_safe_check = all(
        result["safe_distance"] >= collision_radius
        for result in comparison_results
    )
    directional_collision_check = (
        comparison_by_method["Directional CC"]["collision_probability"]
        <= comparison_epsilon + 0.002
    )
    ellipse_collision_check = (
        comparison_by_method["Confidence Ellipse"]["collision_probability"]
        <= comparison_epsilon + 0.002
    )
    directional_less_than_ellipse_check = (
        safe_distance_directional < safe_distance_ellipse
    )
    valid_probabilities_check = all(
        0.0 <= result["collision_probability"] <= 1.0
        for result in comparison_results
    )

    print()
    print("## Three-Method Comparison")
    print()
    print(f"epsilon                  = {comparison_epsilon:.8f}")
    print(f"R                        = {collision_radius:.8f}")
    print(f"alpha                    = {alpha:.8f}")
    print(f"Monte Carlo samples      = {number_of_samples}")
    print(f"random seed              = {random_seed}")
    print()
    print("Heuristic:")
    print(f"  lambda_max             = {lambda_max:.8f}")
    print(f"  sigma_max              = {sigma_max:.8f}")
    print(f"  alpha                  = {alpha:.8f}")
    print("Confidence Ellipse:")
    print(f"  chi2_quantile          = {chi2_quantile:.8f}")
    print(f"  directional radius     = {ellipse_directional_radius:.8f}")
    print("Directional CC:")
    print(f"  beta                   = {comparison_beta:.8f}")
    print(f"  sigma_n                = {comparison_sigma_n:.8f}")
    print()
    print(
        "method                safe_distance   extra_margin   "
        "collision_probability"
    )
    for result in comparison_results:
        print(
            f"{result['method']:<21} "
            f"{result['safe_distance']:<15.8f} "
            f"{result['extra_margin']:<14.8f} "
            f"{result['collision_probability']:.8f}"
        )

    print()
    print("Automatic checks")
    print(
        "1. All safe distances >= R:                "
        f"{check_label(all_distances_safe_check)}"
    )
    print(
        "2. Directional CC collision probability:  "
        f"{check_label(directional_collision_check)}"
    )
    print(
        "3. Confidence ellipse collision safety:   "
        f"{check_label(ellipse_collision_check)}"
    )
    print(
        "4. Directional distance < ellipse distance: "
        f"{check_label(directional_less_than_ellipse_check)}"
    )
    print(
        "5. All probabilities are in [0, 1]:       "
        f"{check_label(valid_probabilities_check)}"
    )


if __name__ == "__main__":
    main()
