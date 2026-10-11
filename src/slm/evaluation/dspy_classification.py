"""Intent classification as a DSPy program, so prompts and few-shot examples can be optimized."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import dspy
import pandas as pd

from slm.evaluation.classification import INVALID


def ollama_lm(model: str, think: bool = False, num_ctx: int = 8192, **kwargs) -> dspy.LM:
    """A DSPy language model served by the local Ollama server (via LiteLLM's ``ollama_chat``)."""
    return dspy.LM(
        f"ollama_chat/{model}",
        api_base="http://localhost:11434",
        api_key="",
        reasoning_effort="medium" if think else "none",  # LiteLLM maps this to Ollama's ``think``
        num_ctx=num_ctx,
        **{"temperature": 0.0, "max_tokens": 4096, **kwargs},
    )


def intent_signature(labels: list[str]) -> type[dspy.Signature]:
    """Signature whose output type is restricted to the label set."""
    return dspy.make_signature(
        {
            "text": (str, dspy.InputField(desc="customer message")),
            "label": (Literal[tuple(labels)], dspy.OutputField(desc="intent of the message")),
        },
        instructions="Classify the intent of a customer message sent to a bank.",
        signature_name="ClassifyIntent",
    )


def to_examples(data: pd.DataFrame) -> list[dspy.Example]:
    return [
        dspy.Example(text=row.text, label=row.label, idx=idx).with_inputs("text")
        for idx, row in data.iterrows()
    ]


def exact_match(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    return example.label == prediction.label


def evaluate_program(
    program: dspy.Module,
    devset: list[dspy.Example],
    num_threads: int = 4,
    output_file: Path | None = None,
) -> dict:
    """Evaluate ``program`` and optionally store per-example predictions as JSONL."""
    evaluator = dspy.Evaluate(
        devset=devset, metric=exact_match, num_threads=num_threads,
        display_progress=True, max_errors=len(devset),
    )
    # bypass DSPy's cache so that wall time measures real inference
    with dspy.context(lm=dspy.settings.lm.copy(cache=False)):
        start = time.perf_counter()
        result = evaluator(program)
        wall_time = time.perf_counter() - start

    records = []
    for example, prediction, correct in result.results:
        label = getattr(prediction, "label", None) or INVALID
        records.append({
            "idx": example.idx,
            "label": example.label,
            "prediction": label,
            "correct": bool(correct),
            "reasoning": getattr(prediction, "reasoning", None),
        })
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("".join(json.dumps(r) + "\n" for r in records))

    predictions = pd.DataFrame(records)
    return {
        "n": len(predictions),
        "accuracy": predictions["correct"].mean(),
        "invalid_rate": (predictions["prediction"] == INVALID).mean(),
        "wall_time_s": wall_time,
    }
