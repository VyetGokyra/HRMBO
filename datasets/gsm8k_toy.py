"""
gsm8k_toy.py
~~~~~~~~~~~~

This module implements a very small toy dataset loosely inspired by the GSM8K
benchmark.  The goal of this dataset is not to test the limits of reasoning but
to provide a simple environment in which to demonstrate the BOPRO + HRM pipeline.

Each problem is a basic arithmetic question involving two integers and one
operation (addition, subtraction or multiplication).  The objective is to
produce the correct answer.  The evaluation function returns `1.0` for a
correct answer and `0.0` otherwise.  The embedding function maps a candidate
solution string to a fixed dimensional vector by hashing the string and using
a pseudo‑random projection.  In a real implementation this embedding would be
obtained from the hidden state of the HRM or a learned encoder.
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import List, Sequence

# Support both package and direct script execution
try:
    # When imported as part of the HRMBO package
    from ..hrm import HierarchicalReasoningModel  # type: ignore
except ImportError:
    # When running experiments directly without package context
    from hrm import HierarchicalReasoningModel  # type: ignore


class GSM8kToyDataset:
    """A tiny dataset of arithmetic word problems for demonstration purposes."""

    def __init__(self, latent_dim: int = 5) -> None:
        # Define a handful of problems
        self.problems: List[dict] = [
            {"question": "What is 2 + 3?", "answer": "5"},
            {"question": "Compute 7 minus 4", "answer": "3"},
            {"question": "What is 5 * 6?", "answer": "30"},
            {"question": "Compute 9 + 8", "answer": "17"},
            {"question": "What is 12 x 3?", "answer": "36"},
        ]
        self.problem_idx = -1
        self.latent_dim = latent_dim
        self.hrm = HierarchicalReasoningModel()

    @property
    def problem_description(self) -> str:
        """Return the question for the current problem."""
        return self.problems[self.problem_idx]["question"]

    def next_problem(self) -> bool:
        """Advance to the next problem.

        Returns ``True`` if there is another problem to solve, ``False`` otherwise.
        """
        if self.problem_idx + 1 >= len(self.problems):
            return False
        self.problem_idx += 1
        return True

    def generate_candidate(self, z: np.ndarray, examples: Sequence[str]) -> str:
        """Generate a candidate answer for the current problem.

        The latent vector and examples are ignored in this toy implementation.  The HRM is
        used directly to produce the answer.

        Parameters
        ----------
        z: np.ndarray
            Latent vector proposed by the acquisition function.  Ignored here.

        examples: Sequence[str]
            A list of example solutions from previous iterations.  Ignored here.

        Returns
        -------
        str
            A candidate answer string.
        """
        question = self.problem_description
        return self.hrm.generate(question, z, examples)

    def evaluate(self, solution: str) -> float:
        """Return a score of 1.0 if the answer matches the ground truth, 0.0 otherwise."""
        correct_answer = self.problems[self.problem_idx]["answer"].strip()
        # Strip any whitespace from both candidate and correct answer
        if solution.strip() == correct_answer:
            return 1.0
        return 0.0

    def embed(self, solution: str) -> np.ndarray:
        """Embed a candidate solution into a latent vector.

        A cryptographic hash (SHA256) of the solution string is computed and used
        to seed a pseudo‑random number generator.  The generator then produces
        ``latent_dim`` floating point numbers in the range [0, 1).  This procedure
        ensures that identical solutions map to the same embedding while different
        solutions map to unrelated points.  In a real application this function
        would be replaced by a learned embedding from the HRM.
        """
        # Compute SHA256 hash of the solution string
        h = hashlib.sha256(solution.encode("utf-8")).hexdigest()
        # Use part of the hash as seed
        seed_int = int(h[:16], 16) % (2**32 - 1)
        rng = np.random.RandomState(seed_int)
        return rng.rand(self.latent_dim)
