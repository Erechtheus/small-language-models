"""Classifying several texts per LLM call (batch prompting)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from slm.evaluation.classification import INVALID, SYSTEM_PROMPT
from slm.inference import OllamaBackend

BATCH_ANSWER_FORMAT = (
    "You will receive several numbered messages. Classify each message on its own and answer with a JSON object "
    'that maps each message number to its intent, e.g. {"1": "<intent>", "2": "<intent>"}.'
)


class BatchClassifier:
    """Classifies a batch of texts in one call.

    The answer is constrained to a JSON object with one key per message number ("1", "2", ...), each restricted
    to the labels, so the model always returns exactly one valid label per message, in a fixed order.
    """

    def __init__(self, backend: OllamaBackend, labels: list[str]):
        self.backend = backend
        self.labels = labels
        self.system_prompt = SYSTEM_PROMPT.format(
            labels="\n".join(f"- {label}" for label in labels), answer_format=BATCH_ANSWER_FORMAT
        )

    def schema(self, n: int) -> dict:
        keys = [str(i) for i in range(1, n + 1)]
        return {
            "type": "object",
            "properties": {key: {"type": "string", "enum": self.labels} for key in keys},
            "required": keys,
        }

    def predict(self, texts: list[str]) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(f"{i}. {text}" for i, text in enumerate(texts, 1))},
        ]
        completion = self.backend.chat(messages, format=self.schema(len(texts)))
        try:
            answer = json.loads(completion.content)
            predictions = [answer.get(str(i), INVALID) for i in range(1, len(texts) + 1)]
        except (json.JSONDecodeError, AttributeError):
            predictions = [INVALID] * len(texts)
        return {
            "predictions": [p if p in self.labels else INVALID for p in predictions],
            "raw": completion.content,
            "latency_s": completion.latency_s,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "done_reason": completion.done_reason,
        }


def make_batches(index: pd.Index, batch_size: int, seed: int = 0) -> list[list[int]]:
    """Shuffle ``index`` and split it into batches (the last one may be smaller)."""
    order = np.random.default_rng(seed).permutation(np.asarray(index))
    return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def evaluate_batches(
    classifier: BatchClassifier,
    data: pd.DataFrame,
    batch_size: int,
    seed: int = 0,
    cache_file: Path | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Classify ``data`` (columns ``text``, ``label``) in random batches; one row per text.

    Each batch is cached as one JSONL record, so an interrupted run resumes where it stopped.
    """
    batches = make_batches(data.index, batch_size, seed)
    done: dict[int, dict] = {}
    if cache_file is not None and cache_file.exists():
        for line in cache_file.read_text().splitlines():
            record = json.loads(line)
            done[record["batch"]] = record
    todo = [b for b in range(len(batches)) if b not in done]
    if todo:
        classifier.backend.ensure_model()

    def run(b: int) -> dict:
        return {"batch": b, "idx": batches[b], **classifier.predict(data.loc[batches[b], "text"].tolist())}

    sink = None
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        sink = cache_file.open("a")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for record in tqdm(pool.map(run, todo), total=len(todo), desc=f"batch size {batch_size}", disable=not todo):
                done[record["batch"]] = record
                if sink is not None:
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
    finally:
        if sink is not None:
            sink.close()

    rows = []
    for b in range(len(batches)):
        record = done[b]
        n = len(record["idx"])
        for position, (idx, prediction) in enumerate(zip(record["idx"], record["predictions"])):
            rows.append({
                "idx": idx,
                "batch": b,
                "position": position,
                "batch_len": n,
                "prediction": prediction,
                # cost of the call, shared equally by the texts in the batch
                "latency_s": record["latency_s"] / n,
                "batch_latency_s": record["latency_s"],
                "prompt_tokens": record["prompt_tokens"] / n,
                "completion_tokens": record["completion_tokens"] / n,
            })
    return data.join(pd.DataFrame(rows).set_index("idx"))
