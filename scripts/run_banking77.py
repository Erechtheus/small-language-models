"""Evaluate a ladder of Qwen models on the BANKING77 test set via Ollama.

Examples:
    uv run python scripts/run_banking77.py --per-class 5
    uv run python scripts/run_banking77.py --models qwen3:0.6b qwen3:1.7b qwen3:4b qwen3:8b
    uv run python scripts/run_banking77.py --mode free --per-class 10
    uv run python scripts/run_banking77.py --think --models qwen3.5:2b qwen3.5:4b --per-class 5
"""

from __future__ import annotations

import argparse

import pandas as pd

from slm import RESULTS_DIR
from slm.data import load_banking77
from slm.evaluation import OUTPUT_MODES, LLMClassifier, evaluate, score, stratified_subset
from slm.inference import THINKING_OPTIONS, OllamaBackend, params_in_billions

DEFAULT_MODELS = ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
OUT_DIR = RESULTS_DIR / "banking77"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--mode", choices=OUTPUT_MODES, default="structured", help="how the output is constrained")
    parser.add_argument("--per-class", type=int, help="evaluate on a balanced subset with this many queries per intent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="parallel requests (see OLLAMA_NUM_PARALLEL)")
    parser.add_argument("--think", action="store_true", help="enable reasoning mode (uses THINKING_OPTIONS sampling)")
    parser.add_argument("--no-pull", action="store_true", help="fail instead of pulling missing models")
    args = parser.parse_args()

    dataset = load_banking77()
    test = dataset.test
    if args.per_class:
        test = stratified_subset(test, args.per_class, seed=args.seed)

    run_name = f"{args.mode}_{'think' if args.think else 'no-think'}"
    options = THINKING_OPTIONS if args.think else {}
    rows = []
    for model in args.models:
        backend = OllamaBackend(model, think=args.think, **options)
        if args.no_pull:
            backend.ensure_model(pull=False)
        classifier = LLMClassifier(backend, dataset.labels, mode=args.mode)

        cache_file = OUT_DIR / run_name / f"{model.replace(':', '_').replace('/', '_')}.jsonl"
        results = evaluate(classifier, test, cache_file=cache_file, workers=args.workers)
        backend.unload()

        metrics = {"model": model, "params_b": params_in_billions(model), **score(results)}
        rows.append(metrics)
        print(f"{model:>20}  acc={metrics['accuracy']:.3f}  lenient={metrics['accuracy_lenient']:.3f}  "
              f"invalid={metrics['invalid_rate']:.1%}")

    summary = pd.DataFrame(rows)
    summary_file = OUT_DIR / f"summary_{run_name}{f'_{args.per_class}pc' if args.per_class else ''}.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\n{summary.to_string(index=False)}\n\nSaved to {summary_file}")


if __name__ == "__main__":
    main()
