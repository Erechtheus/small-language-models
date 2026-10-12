"""Confidence signals for LLM classifiers and how well they are calibrated."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor

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
    steps = label_token_steps(content, logprobs, labels)
    if steps is None:
        return None
    return math.exp(sum(step["log_p"] for step in steps))


def label_token_steps(content: str, logprobs: list[tuple[str, float, list[tuple[str, float]]]],
                      labels: list[str]) -> list[dict] | None:
    """The decisions behind a generated label, one per token (see :func:`label_probability`).

    Each step has the generated ``token``, the ``alternatives`` as (token, logprob, allowed by the schema)
    and ``log_p``, the log-probability of the token renormalized over the allowed alternatives.
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

    steps, pos = [], 0
    for token, logprob, alternatives in logprobs:
        end = pos + len(token)
        if end > start:
            before = text[:pos]
            options = [logprob] + [lp for alt, lp in alternatives if alt and alt != token and valid(before + alt)]
            steps.append({
                "token": token,
                "alternatives": [(alt, lp, bool(alt) and valid(before + alt)) for alt, lp in alternatives],
                "log_p": logprob - _logsumexp(options),
            })
            if end > label_end:
                break
        pos = end
    return steps


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


def label_distribution(client, model: str, prompt: str, labels: list[str], tokenizer, workers: int = 4,
                       ) -> tuple[pd.Series, dict[str, list[tuple[str, float]]]]:
    """Probability of every label as the answer, scored token by token (teacher forcing).

    ``prompt`` is the raw prompt up to the opening quote of the label (``{"label": "``). For every prefix of
    every label, one token is generated with its 20 most likely alternatives; the probability of a label is the
    product of the probabilities of its tokens and of the closing quote, normalized over all labels (which is
    what the JSON schema does). Returns the distribution and the alternatives per prefix.
    """
    tokens = {label: [tokenizer.decode([i]) for i in tokenizer.encode(label, add_special_tokens=False).ids]
              for label in labels}
    prefixes = sorted({"".join(t[:i]) for t in tokens.values() for i in range(len(t) + 1)})

    def top(prefix: str) -> list[tuple[str, float]]:
        response = client.generate(model=model, prompt=prompt + prefix, raw=True, logprobs=True, top_logprobs=20,
                                   options={"temperature": 0.0, "num_predict": 1, "num_ctx": 4096})
        return [(t.token, t.logprob) for t in response.logprobs[0].top_logprobs]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        alternatives = dict(zip(prefixes, pool.map(top, prefixes)))

    scores = {}
    for label, label_tokens in tokens.items():
        p, prefix = 1.0, ""
        for token in label_tokens:
            p *= math.exp(dict(alternatives[prefix]).get(token, -math.inf))
            prefix += token
        p *= sum(math.exp(lp) for alt, lp in alternatives[prefix] if alt.startswith('"'))
        scores[label] = p
    distribution = pd.Series(scores).sort_values(ascending=False)
    return distribution / distribution.sum(), alternatives
