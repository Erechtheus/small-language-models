"""How well do small models know when they are right? Confidence signals on BANKING77.

Signals (all with structured output and thinking off):
    label_prob   probability of the generated label, from the token log-probabilities
                 (same prompt and greedy decoding as the main runs, so the answers are the same)
    stated       the model states its confidence (0-100) in the JSON answer
    sampling     the query is sampled ``--samples`` times; confidence = share of samples that agree
                 with the greedy answer of the main run (needs ``run_banking77.py`` results)

Examples:
    uv run python scripts/run_calibration.py --signal label_prob --models qwen3.5:2b qwen3.5:9b --per-class 5
    uv run python scripts/run_calibration.py --signal label_prob --models qwen3.5:4b
    uv run python scripts/run_calibration.py --signal sampling --models qwen3.5:4b --per-class 5
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from slm import RESULTS_DIR
from slm.data import load_banking77
from slm.evaluation import LLMClassifier, evaluate, stratified_subset
from slm.evaluation.calibration import expected_calibration_error
from slm.inference import OllamaBackend

SIGNALS = ("label_prob", "stated", "sampling")
# Qwen's recommended sampling without thinking
SAMPLING_OPTIONS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
OUT_DIR = RESULTS_DIR / "banking77" / "calibration"


def run_signal(signal: str, model: str, labels: list[str], test: pd.DataFrame, samples: int,
               workers: int) -> pd.DataFrame:
    """Predictions with a ``confidence`` column in [0, 1]."""
    name = model.replace(":", "_")
    if signal == "label_prob":
        backend = OllamaBackend(model, top_logprobs=20)
        results = evaluate(LLMClassifier(backend, labels), test, OUT_DIR / signal / f"{name}.jsonl", workers)
        results["confidence"] = results["label_prob"].fillna(0.0)
    elif signal == "stated":
        backend = OllamaBackend(model)
        results = evaluate(LLMClassifier(backend, labels, confidence=True), test,
                           OUT_DIR / signal / f"{name}.jsonl", workers)
        results["confidence"] = results["stated_confidence"].fillna(0.0) / 100
    else:
        greedy = pd.read_json(RESULTS_DIR / "banking77" / "structured_no-think" / f"{name}.jsonl", lines=True)
        results = test.join(greedy.set_index("idx")[["prediction", "latency_s"]])
        votes = []
        for seed in range(samples):
            backend = OllamaBackend(model, seed=seed, **SAMPLING_OPTIONS)
            sampled = evaluate(LLMClassifier(backend, labels), test,
                               OUT_DIR / signal / name / f"seed{seed}.jsonl", workers)
            votes.append(sampled["prediction"] == results["prediction"])
        results["confidence"] = pd.concat(votes, axis=1).mean(axis=1)
    backend.unload()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"])
    parser.add_argument("--per-class", type=int, help="balanced subset with this many queries per intent")
    parser.add_argument("--samples", type=int, default=10, help="samples per query (signal 'sampling')")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_banking77()
    test = dataset.test if args.per_class is None else stratified_subset(dataset.test, args.per_class)
    suffix = f"_{args.per_class}pc" if args.per_class else ""
    summary_file = OUT_DIR / f"summary_{args.signal}{suffix}.csv"

    rows = []
    for model in args.models:
        results = run_signal(args.signal, model, dataset.labels, test, args.samples, args.workers)
        correct = results["prediction"] == results["label"]
        rows.append({
            "model": model,
            "signal": args.signal,
            "n": len(results),
            "accuracy": correct.mean(),
            "mean_confidence": results["confidence"].mean(),
            "ece": expected_calibration_error(results["confidence"], correct),
            "auroc": roc_auc_score(correct, results["confidence"]),  # does confidence rank right above wrong?
        })
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False))

    summary = pd.DataFrame(rows)
    if summary_file.exists():
        previous = pd.read_csv(summary_file)
        summary = pd.concat([previous[~previous["model"].isin(summary["model"])], summary], ignore_index=True)
    summary.to_csv(summary_file, index=False)
    print(f"\n{summary.round(3).to_string(index=False)}\n\nSaved to {summary_file}")


if __name__ == "__main__":
    main()
