# HRMBO: Combining Hierarchical Reasoning Models with Bayesian Optimisation via Prompting

This repository contains a **minimal, reproducible prototype** of the ideas described in the attached white paper on combining the *Hierarchical Reasoning Model (HRM)* with the *Bayesian Optimisation via Prompting (BOPRO)* algorithm.  The goal of this project is to demonstrate how a small recurrent reasoning system can be used in place of large language models to solve optimisation problems and reason about simple tasks.

## Overview

The original BOPRO algorithm uses a large language model (LLM) to propose candidate solutions during a Bayesian optimisation loop.  A surrogate model (typically a Gaussian process) is fitted on latent representations of previous candidate solutions and their evaluation scores.  An acquisition function is optimised over the latent space to find a new latent point, from which a prompt is constructed and passed to the LLM to generate new candidate solutions.  Those solutions are scored by the problem‐specific objective function, the surrogate is updated and the loop repeats.

In this repository we replace the LLM with a compact **Hierarchical Reasoning Model**.  HRM is a two–tier recurrent architecture inspired by the way the brain performs multi–step reasoning: a slow high–level controller (`H`) plans at a coarse level while a fast low–level controller (`L`) performs detailed computation.  In our simplified implementation the HRM is represented by a small PyTorch module that operates on problem descriptions and latent control vectors and outputs proposed solutions.

Due to the limited resources available in this environment (no access to GPT‑4 or other proprietary APIs and no ability to download large public datasets), the experiments included here are intentionally **toy examples**.  The code is structured to make it straightforward to plug in real datasets (e.g. Semantle, Dockstring, ARC, GSM8K, mathematics datasets, HumanEval) once network access and compute resources are available.  For now, we demonstrate the framework on a small set of arithmetic word problems reminiscent of the GSM8K benchmark.

## Repository Layout

```
HRMBO/
├── bopro.py                # Implementation of the BOPRO optimisation loop
├── hrm.py                  # Simplified hierarchical reasoning model
├── datasets/
│   ├── __init__.py         # Dataset registry
│   └── gsm8k_toy.py        # A tiny arithmetic dataset for demonstration
├── experiments/
│   └── run_experiments.py  # Script to run optimisation experiments and save results
├── results/                # Generated results are stored here
│   └── .gitkeep            # Keeps the directory in git
└── LICENSE
```

### `hrm.py`

Defines a class `HierarchicalReasoningModel` that mimics the high/low level update mechanism of the HRM architecture.  The implementation is deliberately small and contains no learned parameters.  It exposes a `generate()` method that, given a problem description and a latent control vector, returns a candidate solution.  In the toy arithmetic dataset the generator simply parses the question, performs the required calculation and returns the correct answer, illustrating how an HRM might encapsulate reasoning logic internally without emitting intermediate chain–of–thought.

### `bopro.py`

Contains a `BOPROOptimiser` class implementing the Bayesian optimisation loop.  It fits a Gaussian process surrogate model on latent vectors and their corresponding objective scores, uses a simple expected improvement acquisition function to propose new latent points and asks the dataset to generate candidate solutions for those points.  The optimiser is agnostic to the choice of dataset; it requires the dataset to provide methods for embedding solutions into a fixed–dimensional latent space and for generating candidates from latent points.

### `datasets/gsm8k_toy.py`

Implements a small arithmetic dataset with a handful of grade‑school math problems.  It provides:

* `problems`: a list of dictionary objects with a `question` and `answer` field.
* `embed(solution)`: converts a candidate solution into a latent vector by hashing the solution string and projecting it into a fixed dimension.  In a real implementation this would be replaced by a learned embedding or the hidden state of the HRM.
* `generate_candidate(z, examples)`: uses the `HierarchicalReasoningModel` to produce a candidate answer for a given problem.  It ignores the latent vector in this toy example but demonstrates the interface expected by the optimiser.
* `evaluate(solution)`: returns `1.0` if the candidate matches the ground truth answer and `0.0` otherwise.  More fine‐grained scoring functions can easily be substituted.

### `experiments/run_experiments.py`

A command line script that loops through all problems in the arithmetic dataset and runs the BOPRO optimisation loop on each one.  It collects the best solution found and its score, prints the progress to stdout and writes the results to a JSON file in the `results` directory.  To keep runtime reasonable it limits the number of optimisation iterations; these parameters can be adjusted for more thorough exploration.

## Running the Prototype

To reproduce the toy experiments locally, install the dependencies listed in `requirements.txt` (see below), and then run:

```bash
python -m pip install -r requirements.txt
python experiments/run_experiments.py
```

The script will generate a file `results/gsm8k_toy_results.json` containing the outcome of the optimisation runs.  Because the toy HRM always returns the correct answer, the optimisation converges immediately; nonetheless the infrastructure for Bayesian optimisation, surrogate fitting and candidate generation is fully functional.

## Limitations & Future Work

This repository is **not** a full re‑implementation of the large scale experiments described in the white paper.  In particular:

* The HRM implemented here is a hand‑crafted stub rather than a trained model.  Training an HRM requires data and compute that are unavailable in this environment.
* We do not provide code for or run experiments on Semantle, Dockstring, ARC, GSM8K, DeepMind Mathematics, HumanEval or APPS.  The structure of the `datasets` package makes it straightforward to add these datasets when they can be downloaded (e.g. via the `datasets` library), and to implement appropriate evaluation functions.
* The optimisation loop uses a very simple acquisition strategy and latent space; more sophisticated techniques (e.g. gradient–based acquisition optimisation, acquisition functions like UCB or Thompson sampling) can be plugged in with minor changes.

Despite these limitations, the code offers a solid starting point for experimenting with HRM‑augmented Bayesian optimisation on a variety of tasks.  Extending the system to real datasets is left as future work.

## Requirements

The prototype relies on a small number of Python packages:

```
numpy
scikit-learn
torch
```

`requirements.txt` is intentionally minimal to ensure compatibility with resource constrained environments.  Additional packages such as `datasets`, `rdkit`, `einops` etc. will be required when extending to real tasks (e.g. molecule optimisation).

## License

This project is licensed under the Apache 2.0 License.  See the [LICENSE](LICENSE) file for details.
