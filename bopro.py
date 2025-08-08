"""
bopro.py
~~~~~~~~

This module implements a simple version of the Bayesian optimisation via prompting (BOPRO)
algorithm described in the accompanying white paper.  The optimiser operates over a latent
space of fixed dimension.  It maintains a Gaussian process surrogate model over previously
evaluated points and uses an acquisition function to propose new latent points.  These latent
points are then decoded into candidate solutions by the dataset's `generate_candidate` method
(which wraps a hierarchical reasoning model in our experiments).  After evaluation, the
surrogate model is updated and the process repeats for a specified number of iterations.

The implementation here is deliberately lightweight: we use scikit‑learn's
``GaussianProcessRegressor`` with a squared exponential kernel and a basic expected improvement
acquisition function.  More sophisticated kernels and acquisition strategies can be integrated
without changing the overall interface.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, List, Sequence, Tuple, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class BOPROOptimiser:
    """Bayesian optimisation via prompting.

    Parameters
    ----------
    dataset : object
        An object that defines the problem domain.  It must implement methods:

        * ``embed(solution: str | any) -> np.ndarray``: embeds a candidate solution into a latent
          vector of dimension ``latent_dim``.
        * ``generate_candidate(z: np.ndarray, examples: Sequence[str]) -> str``: given a latent
          vector proposed by the acquisition function and a list of example solutions, produce
          a new candidate solution.  This will typically call into a hierarchical reasoning
          model.
        * ``evaluate(solution: str | any) -> float``: compute the objective score for a candidate.
        * ``problem_description``: a string describing the current problem (used by
          ``generate_candidate``).

    latent_dim : int
        Dimensionality of the latent space.  Must match the dimension returned by the dataset's
        ``embed`` method.

    k_neighbors : int
        Number of nearest neighbours in the latent space to pass as examples to
        ``generate_candidate``.  These are chosen based on Euclidean distance in latent space.

    num_iterations : int
        Number of optimisation iterations to perform per problem.

    random_state : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: object,
        latent_dim: int,
        k_neighbors: int = 3,
        num_iterations: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.k_neighbors = k_neighbors
        self.num_iterations = num_iterations
        self.random_state = np.random.RandomState(random_state)

        # Surrogate model: GP with RBF kernel + small noise term
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        # Storage for observed data
        self.zs: List[np.ndarray] = []
        self.scores: List[float] = []
        self.solutions: List[str] = []

    def _expected_improvement(self, z: np.ndarray, xi: float = 0.01) -> float:
        """Compute the expected improvement acquisition value for a latent point.

        Parameters
        ----------
        z : np.ndarray
            A 1D array representing a latent point at which to evaluate the acquisition function.

        xi : float
            Exploration parameter: larger values favour exploration.

        Returns
        -------
        float
            The expected improvement at ``z``.
        """
        if len(self.scores) == 0:
            return 0.0
        # Predict mean and standard deviation at z
        mu, sigma = self.gp.predict(z.reshape(1, -1), return_std=True)
        mu = mu.item()
        sigma = sigma.item()
        if sigma < 1e-9:
            return 0.0
        best = max(self.scores)
        improvement = mu - best - xi
        z_score = improvement / sigma
        from scipy.stats import norm  # imported lazily to avoid dependency unless needed
        ei = improvement * norm.cdf(z_score) + sigma * norm.pdf(z_score)
        return float(ei)

    def _select_examples(self, z: np.ndarray) -> List[str]:
        """Select the nearest previously evaluated solutions to ``z`` for prompting.

        Returns the ``k_neighbors`` solutions with smallest Euclidean distance in latent space.
        If fewer than ``k_neighbors`` solutions are available, all are returned.
        """
        if len(self.zs) == 0 or self.k_neighbors <= 0:
            return []
        # Compute distances
        distances = [np.linalg.norm(existing_z - z) for existing_z in self.zs]
        # Get indices of the smallest distances
        indices = np.argsort(distances)[: self.k_neighbors]
        return [self.solutions[i] for i in indices]

    def optimise(self) -> Tuple[str, float]:
        """Perform Bayesian optimisation for a single problem.

        Returns
        -------
        Tuple[str, float]
            A tuple containing the best solution found and its score.
        """
        # Reset state for each problem
        self.zs.clear()
        self.scores.clear()
        self.solutions.clear()

        # Main optimisation loop
        for iteration in range(self.num_iterations):
            # If we have enough data, fit the GP; otherwise skip fitting
            if len(self.scores) > 0:
                X = np.stack(self.zs)
                y = np.array(self.scores)
                # Catch potential numerical issues
                try:
                    self.gp.fit(X, y)
                except Exception:
                    # In the unlikely event of failure, skip updating the GP
                    pass
            # Propose next latent point
            if len(self.scores) == 0:
                # Randomly sample initial latent point from standard normal
                z_next = self.random_state.randn(self.latent_dim)
            else:
                # Sample several random points and pick the one with highest acquisition
                num_candidates = 100
                candidate_z = self.random_state.randn(num_candidates, self.latent_dim)
                ei_vals = [self._expected_improvement(z) for z in candidate_z]
                best_idx = int(np.argmax(ei_vals))
                z_next = candidate_z[best_idx]
            # Select example solutions for prompting
            examples = self._select_examples(z_next)
            # Ask dataset to generate a candidate solution
            solution = self.dataset.generate_candidate(z_next, examples)
            # Evaluate the candidate and embed it into latent space
            score = float(self.dataset.evaluate(solution))
            z_emb = self.dataset.embed(solution)
            # Store the observation
            self.zs.append(z_emb)
            self.scores.append(score)
            self.solutions.append(solution)
            # Early stopping: if we've achieved the maximum possible score, break
            if score >= 1.0:
                break
        # Return the best found solution
        if len(self.scores) == 0:
            return "", 0.0
        best_idx = int(np.argmax(self.scores))
        return self.solutions[best_idx], self.scores[best_idx]
