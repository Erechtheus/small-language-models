"""Zero-shot text classification with an LLM, with different levels of output control."""

from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from slm.evaluation.calibration import label_probability
from slm.inference import OllamaBackend

INVALID = "<invalid>"

OutputMode = Literal["structured", "json", "free"]
OUTPUT_MODES: tuple[OutputMode, ...] = ("structured", "json", "free")

SYSTEM_PROMPT = """You are an intent classifier for customer messages sent to a bank.
Assign the message to exactly one of the following intents:

{labels}

{answer_format}"""

ANSWER_FORMATS: dict[OutputMode, str] = {
    "structured": 'Answer with a JSON object of the form {"label": "<intent>"}.',
    "json": 'Answer with a JSON object of the form {"label": "<intent>"}.',
    "free": "Answer with the name of the intent only, exactly as written in the list above.",
}

# asking for a verbalized confidence along with the label (structured and json mode)
CONFIDENCE_FORMAT = (
    'Answer with a JSON object of the form {"label": "<intent>", "confidence": <0-100>}, where confidence '
    "is the probability in percent that your answer is correct."
)

# asking for a short explanation before or after the label (structured mode)
Explanation = Literal["before", "after"]
EXPLANATION_FORMATS: dict[Explanation, str] = {
    "before": (
        'Answer with a JSON object of the form {"reason": "<one or two sentences>", "label": "<intent>"}: first '
        "explain briefly which intent fits the message and why, then give the intent."
    ),
    "after": (
        'Answer with a JSON object of the form {"label": "<intent>", "reason": "<one or two sentences>"}: first '
        "give the intent, then explain briefly why it fits the message."
    ),
}
MAX_REASON_CHARS = 300


class LLMClassifier:
    """Classifies a text into one of a fixed label set.

    ``mode`` controls how much the output is constrained:

    * ``structured``: JSON schema with an ``enum`` of the labels (constrained decoding), so only
      valid labels can be generated.
    * ``json``: the output is guaranteed to be valid JSON, but the label inside is free text.
    * ``free``: no constraint at all; the answer is plain text and has to be mapped to a label.

    With ``confidence=True`` the model also states how likely its answer is correct (0-100).
    With ``explanation="before"`` the model writes a short reason before the label (a small chain of thought
    inside the JSON); with ``"after"`` the label is already fixed when the reason is written.
    If the backend returns token log-probabilities, the probability of the generated label is
    recorded as ``label_prob``.
    """

    def __init__(self, backend: OllamaBackend, labels: list[str], mode: OutputMode = "structured",
                 confidence: bool = False, explanation: Explanation | None = None):
        if confidence and mode == "free":
            raise ValueError("a stated confidence needs a JSON answer (structured or json mode)")
        if explanation and (mode != "structured" or confidence):
            raise ValueError("an explanation is only supported in structured mode without confidence")
        self.backend = backend
        self.labels = labels
        self.mode = mode
        self.confidence = confidence
        self.explanation = explanation
        answer_format = ANSWER_FORMATS[mode]
        if confidence:
            answer_format = CONFIDENCE_FORMAT
        elif explanation:
            answer_format = EXPLANATION_FORMATS[explanation]
        self.system_prompt = SYSTEM_PROMPT.format(
            labels="\n".join(f"- {label}" for label in labels), answer_format=answer_format
        )
        self.format = {
            "structured": {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": labels}},
                "required": ["label"],
            },
            "json": "json",
            "free": None,
        }[mode]
        if confidence and mode == "structured":
            self.format["properties"]["confidence"] = {"type": "integer", "minimum": 0, "maximum": 100}
            self.format["required"].append("confidence")
        if explanation:
            # the schema fixes the order of the keys, and with it whether the reason can influence the label
            reason = {"type": "string", "maxLength": MAX_REASON_CHARS}
            label = self.format["properties"]["label"]
            order = ["reason", "label"] if explanation == "before" else ["label", "reason"]
            self.format["properties"] = {key: {"reason": reason, "label": label}[key] for key in order}
            self.format["required"] = order

    def predict(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        completion = self.backend.chat(messages, format=self.format)
        strict, lenient = self.parse(completion.content)
        record = {
            "prediction": strict,
            "prediction_lenient": lenient,
            "raw": completion.content,
            "latency_s": completion.latency_s,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "done_reason": completion.done_reason,
        }
        if completion.thinking:
            record["thinking"] = completion.thinking
        if self.confidence:
            record["stated_confidence"] = self._stated_confidence(completion.content)
        if completion.logprobs and strict != INVALID:
            record["label_prob"] = label_probability(completion.content, completion.logprobs, self.labels)
        return record

    @staticmethod
    def _stated_confidence(content: str) -> float | None:
        try:
            value = json.loads(content)["confidence"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        return float(value) if isinstance(value, (int, float)) else None

    def parse(self, content: str) -> tuple[str, str]:
        """Map the raw model output to a label, once strictly and once leniently.

        Strict: the answer must be exactly one of the labels.
        Lenient: case, quotes, spaces vs. underscores and small typos are forgiven, and a label
        mentioned anywhere in a longer answer is accepted.
        """
        answer = content.strip()
        if self.mode in ("structured", "json"):
            try:
                answer = json.loads(answer)["label"]
            except (json.JSONDecodeError, KeyError, TypeError):
                return INVALID, INVALID
            if not isinstance(answer, str):
                return INVALID, INVALID
        strict = answer if answer in self.labels else INVALID
        return strict, self._match_leniently(answer)

    def _match_leniently(self, answer: str) -> str:
        normalized = re.sub(r"[\s\-]+", "_", answer.strip().strip("`*\"'.:-").strip().lower())
        if normalized in self.labels:
            return normalized
        # a label mentioned inside a longer answer ("The intent is card_arrival.")
        mentioned = [label for label in self.labels if label in normalized]
        if mentioned:
            return max(mentioned, key=len)
        close = difflib.get_close_matches(normalized, self.labels, n=1, cutoff=0.8)
        return close[0] if close else INVALID


def evaluate(
    classifier: LLMClassifier,
    data: pd.DataFrame,
    cache_file: Path | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Run ``classifier`` over ``data`` (columns ``text``, ``label``).

    Predictions are appended to ``cache_file`` (JSONL) as they arrive, so an
    interrupted run resumes where it stopped. The model is only loaded if something is left to do.
    """
    done: dict[int, dict] = {}
    if cache_file is not None and cache_file.exists():
        for line in cache_file.read_text().splitlines():
            record = json.loads(line)
            done[record["idx"]] = record

    todo = [(idx, row.text) for idx, row in data.iterrows() if idx not in done]
    if todo:
        classifier.backend.ensure_model()

    def run(item: tuple[int, str]) -> dict:
        idx, text = item
        return {"idx": idx, **classifier.predict(text)}

    sink = None
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        sink = cache_file.open("a")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for record in tqdm(pool.map(run, todo), total=len(todo), desc=classifier.backend.model):
                done[record["idx"]] = record
                if sink is not None:
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
    finally:
        if sink is not None:
            sink.close()

    predictions = pd.DataFrame([done[idx] for idx in data.index]).set_index("idx")
    results = data.join(predictions)
    # caches written before lenient parsing existed (structured mode, where both are identical)
    if "prediction_lenient" not in results:
        results["prediction_lenient"] = results["prediction"]
    results["prediction_lenient"] = results["prediction_lenient"].fillna(results["prediction"])
    return results


def stratified_subset(data: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """``per_class`` random examples of every label: a small but balanced evaluation set."""
    return data.groupby("label", group_keys=False).sample(n=per_class, random_state=seed).sort_index()


def score(results: pd.DataFrame) -> dict:
    """Quality and cost metrics for the output of :func:`evaluate`."""
    metrics = {
        "n": len(results),
        "accuracy": accuracy_score(results["label"], results["prediction"]),
        "macro_f1": f1_score(results["label"], results["prediction"], average="macro", zero_division=0),
        "invalid_rate": (results["prediction"] == INVALID).mean(),
        "accuracy_lenient": accuracy_score(results["label"], results["prediction_lenient"]),
        "invalid_rate_lenient": (results["prediction_lenient"] == INVALID).mean(),
        "mean_latency_s": results["latency_s"].mean(),
        "p95_latency_s": results["latency_s"].quantile(0.95),
        "mean_prompt_tokens": results["prompt_tokens"].mean(),
        "mean_completion_tokens": results["completion_tokens"].mean(),
    }
    if "done_reason" in results:
        metrics["truncated_rate"] = (results["done_reason"] == "length").mean()
    return metrics
