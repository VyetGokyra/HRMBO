"""
hrm.py
~~~~~~~~

This module defines a very small and simplified **Hierarchical Reasoning Model (HRM)**.  The goal of the
HRM in this project is not to be state‑of‑the‑art but to provide a clean interface that mimics the
behaviour of a trained HRM.  In the full research setting an HRM would be a neural network with
millions of parameters capable of performing multi‑step reasoning.  Here we provide a deterministic
implementation that can parse simple arithmetic questions and return the correct answer.

The HRM exposes a single public method `generate(problem, z=None, examples=None)` which accepts a
problem description (a string), an optional latent vector `z` produced by the acquisition function and
an optional set of example solutions.  In a proper HRM these inputs would influence the internal
reasoning process.  In our toy version only the problem string is used; the latent vector is ignored.

The class is intentionally lightweight and free of external dependencies beyond the Python standard
library.  It can be replaced with a learned PyTorch implementation without changing the external
interface.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence


class HierarchicalReasoningModel:
    """A toy hierarchical reasoning model for simple arithmetic problems.

    This model pretends to perform sophisticated reasoning but in reality it simply
    parses the question and computes the result.  The interface is designed to
    mimic that of a more complex HRM so that it can be swapped out later.
    """

    def __init__(self) -> None:
        # In a real HRM there would be parameters here.  We keep this empty.
        pass

    def _parse_and_compute(self, problem: str) -> Optional[str]:
        """Parse a simple arithmetic problem and return the answer as a string.

        The parser handles addition, subtraction and multiplication of two integers.
        If the pattern is not recognised, ``None`` is returned.

        Parameters
        ----------
        problem: str
            A string containing a simple arithmetic question such as
            "What is 3 + 5?" or "Compute 7 minus 2".

        Returns
        -------
        Optional[str]
            The computed answer as a string or ``None`` if the problem could not be parsed.
        """
        # Normalise the string to lower case and remove commas
        s = problem.lower().strip().replace(",", "")
        # Look for addition
        add_match = re.search(r"(\d+)\s*\+\s*(\d+)", s)
        if add_match:
            a, b = map(int, add_match.groups())
            return str(a + b)
        # Look for minus / subtraction
        sub_match = re.search(r"(\d+)\s*(?:-\s*|minus\s+)(\d+)", s)
        if sub_match:
            a, b = map(int, sub_match.groups())
            return str(a - b)
        # Look for multiplication ("*", "x", or "times")
        mul_match = re.search(r"(\d+)\s*(?:\*|x|×|times)\s*(\d+)", s)
        if mul_match:
            a, b = map(int, mul_match.groups())
            return str(a * b)
        return None

    def generate(
        self,
        problem: str,
        z: Optional[Sequence[float]] = None,
        examples: Optional[Sequence[str]] = None,
    ) -> str:
        """Generate a candidate solution for a given problem.

        Parameters
        ----------
        problem: str
            The problem description.  In a real HRM this would be tokenised input
            suitable for the model.  Here it is a raw string.

        z: Sequence[float], optional
            A latent control vector proposed by the acquisition function.  It can
            influence the generation in a real model.  In this toy version it is
            ignored.

        examples: Sequence[str], optional
            Example solutions from previous iterations.  These are used in
            prompting large language models but are unused here.  Provided for
            API completeness.

        Returns
        -------
        str
            The candidate solution.  For recognised arithmetic problems this will
            be the correct answer; otherwise a placeholder string is returned.
        """
        answer = self._parse_and_compute(problem)
        if answer is not None:
            return answer
        # Fallback: if we can't parse, return an empty string
        return ""
