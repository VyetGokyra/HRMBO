"""
run_experiments.py
~~~~~~~~~~~~~~~~~~

This script demonstrates the use of the BOPRO + HRM pipeline on a simple arithmetic dataset.
It iterates over each problem in the ``GSM8kToyDataset``, invokes the Bayesian optimiser to
search for an answer and records the result.  Because the HRM in this toy example is
deterministic and always produces the correct answer, the optimiser converges immediately.
Nevertheless, this script exercises all of the components: latent space sampling, surrogate
fitting, acquisition maximisation, candidate generation and evaluation.

The results are saved to ``results/gsm8k_toy_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

# Ensure that the package root is on the Python path when running this script directly.
script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(script_dir)
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from bopro import BOPROOptimiser  # type: ignore
from datasets import GSM8kToyDataset  # type: ignore


def main() -> None:
    # Create the results directory if it does not exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    # Instantiate the dataset
    latent_dim = 5
    dataset = GSM8kToyDataset(latent_dim=latent_dim)
    # Prepare a container for results
    results: List[Dict] = []
    # Loop over problems
    problem_idx = 0
    while dataset.next_problem():
        # Create optimiser for this problem
        optimiser = BOPROOptimiser(
            dataset=dataset,
            latent_dim=latent_dim,
            k_neighbors=2,
            num_iterations=10,
            random_state=42,
        )
        # Solve the problem
        best_solution, best_score = optimiser.optimise()
        # Record result
        results.append(
            {
                "problem_index": problem_idx,
                "question": dataset.problem_description,
                "best_solution": best_solution,
                "score": best_score,
            }
        )
        print(f"Problem {problem_idx}: {dataset.problem_description}")
        print(f"  Best solution: {best_solution}")
        print(f"  Score: {best_score}\n")
        problem_idx += 1
    # Save results to JSON
    results_path = os.path.join(results_dir, "gsm8k_toy_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
