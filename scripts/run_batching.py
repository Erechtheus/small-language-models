"""Batch prompting on BANKING77: classify several queries per call.

The queries of the balanced subset are shuffled and split into batches; every batch is one call with a
numbered list of queries, and the answer is constrained to one valid label per query.

Examples:
    uv run python scripts/run_batching.py
    uv run python scripts/run_batching.py --models qwen3.5:4b --batch-sizes 1 10 40 --seeds 0 1
"""

from __future__ import annotations

import argparse

import pandas as pd

from slm import RESULTS_DIR
from slm.data import load_banking77
from slm.evaluation import stratified_subset
from slm.evaluation.batch_classification import BatchClassifier, evaluate_batches
from slm.inference import OllamaBackend

OUT_DIR = RESULTS_DIR / "banking77" / "batching"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["qwen3.5:2b"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="random batch compositions")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_banking77()
    test = stratified_subset(dataset.test, args.per_class)
    summary_file = OUT_DIR / f"summary_{args.per_class}pc.csv"

    rows = []
    for model in args.models:
        # room for the label list, 40 queries and their answers
        backend = OllamaBackend(model, num_ctx=8192)
        classifier = BatchClassifier(backend, dataset.labels)
        for batch_size in args.batch_sizes:
            for seed in args.seeds:
                cache = OUT_DIR / model.replace(":", "_") / f"bs{batch_size}_seed{seed}.jsonl"
                results = evaluate_batches(classifier, test, batch_size, seed, cache, args.workers)
                correct = results["prediction"] == results["label"]
                rows.append({
                    "model": model,
                    "batch_size": batch_size,
                    "seed": seed,
                    "accuracy": correct.mean(),
                    "invalid_rate": (results["prediction"] == "<invalid>").mean(),
                    "latency_per_query_s": results["latency_s"].mean(),
                    "prompt_tokens_per_query": results["prompt_tokens"].mean(),
                })
                print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False, header=len(rows) == 1))
        backend.unload()

    summary = pd.DataFrame(rows)
    if summary_file.exists():
        previous = pd.read_csv(summary_file)
        keys = ["model", "batch_size", "seed"]
        previous = previous[~previous.set_index(keys).index.isin(summary.set_index(keys).index)]
        summary = pd.concat([previous, summary], ignore_index=True)
    summary.to_csv(summary_file, index=False)
    print(f"\n{summary.round(3).to_string(index=False)}\n\nSaved to {summary_file}")


if __name__ == "__main__":
    main()
