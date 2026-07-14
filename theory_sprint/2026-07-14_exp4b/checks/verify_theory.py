#!/usr/bin/env python3
"""Numerically red-team the sprint's matrix claims.

Falsifiers: a failed assertion kills the corresponding universal theorem. The
noncommuting and relative-ratio sections instead construct counterexamples to
the naive weighted-ranking and unbiased-relative-loss claims.
"""

from __future__ import annotations

import math

import numpy as np


def hat(x: np.ndarray, tau: float) -> np.ndarray:
    gram = x @ x.T
    return gram @ np.linalg.inv(gram + tau * np.eye(len(x)))


def representation_from_hat(k: np.ndarray, tau: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(k)
    positive = values > 1e-12
    scales = np.sqrt(tau * values[positive] / (1 - values[positive]))
    return vectors[:, positive] * scales


def check_zero_set_and_nonconverse() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    z = x @ q
    assert np.allclose(x @ x.T, z @ z.T, atol=1e-12)
    assert np.allclose(hat(x, 2.3), hat(z, 2.3), atol=1e-12)
    assert np.linalg.norm(x - z) > 1.0


def check_activation_bound() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        x = rng.normal(size=(7, 4))
        z = rng.normal(size=(7, 4))
        tau = float(np.exp(rng.uniform(-2, 2)))
        lhs = np.linalg.norm(hat(x, tau) - hat(z, tau), "fro")
        rhs = (
            (np.linalg.norm(x, 2) + np.linalg.norm(z, 2))
            * np.linalg.norm(x - z, "fro")
            / tau
        )
        assert lhs <= rhs + 1e-10


def check_spectral_tail() -> None:
    rng = np.random.default_rng(2)
    for n, d in ((8, 5), (5, 8)):
        x = rng.normal(size=(n, d))
        u, singular, vh = np.linalg.svd(x, full_matrices=False)
        tau = 1.7
        target = hat(x, tau)
        for rank in range(len(singular) + 1):
            truncated = (u[:, :rank] * singular[:rank]) @ vh[:rank]
            observed = np.linalg.norm(target - hat(truncated, tau), "fro") ** 2
            gains = singular**2 / (singular**2 + tau)
            predicted = np.sum(gains[rank:] ** 2)
            assert np.allclose(observed, predicted, atol=1e-10)


def weighted_objective(k: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> float:
    error = k - b
    return float(np.trace(error @ sigma @ error))


def noncommuting_counterexample() -> dict[str, float]:
    # K and Sigma do not commute. A rank-one attainable prediction operator
    # rotated away from both coordinate eigenvectors beats either naive mode.
    k = np.diag([0.9, 0.2])
    sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    assert not np.allclose(k @ sigma, sigma @ k)
    best = (math.inf, 0.0, 0.0)
    for theta in np.linspace(0, math.pi, 200_001):
        v = np.array([math.cos(theta), math.sin(theta)])
        numerator = float(v @ ((sigma @ k + k @ sigma) / 2) @ v)
        denominator = float(v @ sigma @ v)
        alpha = float(np.clip(numerator / denominator, 0.0, 1 - 1e-12))
        b = alpha * np.outer(v, v)
        value = weighted_objective(k, b, sigma)
        if value < best[0]:
            best = (value, theta, alpha)
    coordinate = min(
        weighted_objective(k, k[i, i] * np.eye(2)[[i]].T @ np.eye(2)[[i]], sigma)
        for i in range(2)
    )
    assert best[0] < coordinate - 1e-5
    return {
        "best_objective": best[0],
        "best_theta_degrees": best[1] * 180 / math.pi,
        "best_hat_eigenvalue": best[2],
        "best_coordinate_objective": coordinate,
    }


def exact_noncommuting_counterexample() -> None:
    # Rational construction promoted to the final theory.
    k = np.diag([4 / 5, 1 / 5])
    sigma = np.array([[11 / 20, 1 / 2], [1 / 2, 11 / 20]])
    v = np.array([4.0, 1.0]) / np.sqrt(17)
    rotated = (61 / 89) * np.outer(v, v)
    rotated_cost = weighted_objective(k, rotated, sigma)
    coordinate_costs = []
    for index in range(2):
        direction = np.eye(2)[:, index]
        coordinate = k[index, index] * np.outer(direction, direction)
        coordinate_costs.append(weighted_objective(k, coordinate, sigma))
    assert np.allclose(rotated_cost, 964 / 189125, atol=1e-14)
    assert rotated_cost < min(coordinate_costs)


def grouping_counterexample() -> None:
    x = np.ones((4, 1))
    z = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    tau = 1.0

    def relative_loss(groups: tuple[tuple[int, int], tuple[int, int]]) -> float:
        numerator = 0.0
        denominator = 0.0
        for group in groups:
            indices = np.asarray(group)
            original = hat(x[indices], tau)
            reconstructed = hat(z[indices], tau)
            numerator += np.linalg.norm(original - reconstructed, "fro") ** 2
            denominator += np.linalg.norm(original, "fro") ** 2
        return numerator / denominator

    assert np.allclose(relative_loss(((0, 1), (2, 3))), 0.0)
    assert np.allclose(relative_loss(((0, 2), (1, 3))), 2.0)


def relative_ratio_bias_counterexample() -> dict[str, float]:
    # Both diagonal matrices are attainable ridge hat matrices. The numerator
    # and denominator are unbiased trace estimates separately, but their ratio
    # is not an unbiased estimate of the ratio of traces.
    k = np.diag([0.9, 0.5])
    khat = np.diag([0.2, 0.4])
    a = k - khat
    exact = np.trace(a @ a) / np.trace(k @ k)
    rng = np.random.default_rng(3)
    y = rng.normal(size=(1_000_000, 2))
    y *= np.sqrt(2 / np.sum(y * y, axis=1, keepdims=True))
    numerator = np.sum((y @ a.T) ** 2, axis=1)
    denominator = np.sum((y @ k.T) ** 2, axis=1)
    estimated_expectation = float(np.mean(numerator / denominator))
    assert abs(estimated_expectation - exact) > 1e-2
    # Confirm attainability, rather than using arbitrary contractions.
    assert np.allclose(hat(representation_from_hat(k, 1.0), 1.0), k)
    assert np.allclose(hat(representation_from_hat(khat, 1.0), 1.0), khat)
    return {
        "ratio_of_traces": float(exact),
        "mean_single_probe_ratio": estimated_expectation,
        "bias": estimated_expectation - float(exact),
    }


def fixed_radius_variance_check() -> dict[str, float]:
    rng = np.random.default_rng(4)
    n = 6
    m = 8
    matrix = rng.normal(size=(n, n))
    b = matrix.T @ matrix
    trials = 200_000
    y = rng.normal(size=(trials, m, n))
    y *= np.sqrt(n / np.sum(y * y, axis=2, keepdims=True))
    estimates = np.einsum("tmi,ij,tmj->tm", y, b, y).mean(axis=1)
    trace = np.trace(b)
    trace2 = np.trace(b @ b)
    predicted = 2 * (n * trace2 - trace**2) / (m * (n + 2))
    observed = float(np.var(estimates))
    assert abs(observed / predicted - 1) < 0.02
    return {"observed_variance": observed, "predicted_variance": float(predicted)}


def fisher_separation_examples() -> None:
    # Equal K but a frozen coordinate readout changes under feature rotation.
    x = np.eye(2)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = x @ swap
    assert np.allclose(hat(x, 1.0), hat(z, 1.0))
    frozen_weight = np.array([1.0, 0.0])
    assert not np.allclose(x @ frozen_weight, z @ frozen_weight)

    # A frozen map that reads only coordinate one has zero pullback length for
    # a coordinate-two perturbation, although the ridge operator can change.
    x2 = np.array([[1.0, 0.0], [0.0, 0.0]])
    z2 = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert not np.allclose(hat(x2, 1.0), hat(z2, 1.0))
    delta = z2 - x2
    assert np.allclose(delta @ frozen_weight, 0.0)


def main() -> None:
    check_zero_set_and_nonconverse()
    check_activation_bound()
    check_spectral_tail()
    exact_noncommuting_counterexample()
    grouping_counterexample()
    fisher_separation_examples()
    print("noncommuting", noncommuting_counterexample())
    print("relative_ratio", relative_ratio_bias_counterexample())
    print("fixed_radius", fixed_radius_variance_check())
    print("all universal checks passed")


if __name__ == "__main__":
    main()
