"""Confidence signals for LLM classifiers and how well they are calibrated."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def label_probability(content: str, logprobs: list[tuple[str, float, list[tuple[str, float]]]],
                      labels: list[str]) -> float | None:
    """Probability of the generated label, from the token log-probabilities of a structured answer.

    Ollama reports the model's distribution *before* the JSON schema is applied, so part of the
    probability mass falls on tokens the schema forbids. For every token of the label we therefore
    renormalize over the alternatives that still lead to a valid label, and multiply the results.
    Only the 20 most likely alternatives are known, so the probability is slightly overestimated
    when valid alternatives are rarer than that.
    """
    text = "".join(token for token, _, _ in logprobs)
    # with reasoning models, the tokens also cover the reasoning trace before the answer
    offset = max(text.rfind(content), 0)
    match = re.search(r'"label"\s*:\s*"', text[offset:])
    if match is None:
        return None
    start = offset + match.end()
    label_end = text.find('"', start)  # the closing quote ends the decision
    targets = [text[:start] + label + '"' for label in labels]

    def valid(prefix: str) -> bool:
        return any(target.startswith(prefix) or prefix.startswith(target) for target in targets)

    log_p, pos = 0.0, 0
    for token, logprob, alternatives in logprobs:
        end = pos + len(token)
        if end > start:
            before = text[:pos]
            options = [logprob] + [lp for alt, lp in alternatives if alt and alt != token and valid(before + alt)]
            log_p += logprob - _logsumexp(options)
            if end > label_end:
                break
        pos = end
    return math.exp(log_p)


def _logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def reliability_table(confidence: pd.Series, correct: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Accuracy per confidence bin: for a calibrated model, accuracy equals mean confidence."""
    edges = np.linspace(0, 1, bins + 1)
    bin_ = pd.cut(confidence, edges, include_lowest=True)
    table = pd.DataFrame({"confidence": confidence, "correct": correct.astype(float), "bin": bin_})
    return table.groupby("bin", observed=True).agg(
        mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size")
    ).reset_index()


def expected_calibration_error(confidence: pd.Series, correct: pd.Series, bins: int = 10) -> float:
    """Average gap between confidence and accuracy over the bins, weighted by bin size."""
    table = reliability_table(confidence, correct, bins)
    return float((table["n"] * (table["mean_confidence"] - table["accuracy"]).abs()).sum() / table["n"].sum())
