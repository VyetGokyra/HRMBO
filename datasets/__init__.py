"""
datasets package
================

This package exposes datasets used by the HRMBO experiments.  Each dataset module must
implement a `Dataset` class with the following interface:

* `problem_description`: a string describing the current optimisation problem.
* `generate_candidate(z: np.ndarray, examples: Sequence[str]) -> str`: given a latent point
  proposed by the acquisition function and a list of example solutions, generate a new
  candidate solution.  Internally this will call the `HierarchicalReasoningModel` to
  produce a candidate.
* `evaluate(solution: str | any) -> float`: compute the objective score for a candidate.
* `embed(solution: str | any) -> np.ndarray`: embed the solution into a fixed dimensional
  latent vector used by the surrogate model.

Additional convenience methods and properties can be added as needed.  See
``datasets.gsm8k_toy.GSM8kToyDataset`` for a concrete example.
"""

from __future__ import annotations

from .gsm8k_toy import GSM8kToyDataset  # noqa: F401

__all__ = ["GSM8kToyDataset"]
