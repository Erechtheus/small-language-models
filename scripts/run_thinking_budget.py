"""Accuracy vs. thinking budget on BANKING77, from the traces of a ``--think`` run.

For every query of the balanced subset:
1. traces that hit the original limit (4,096 tokens) are continued by up to ``--extend`` tokens;
2. traces that end during the extension are closed and answered naturally;
3. for every budget smaller than a trace's natural length, the trace is cut at the budget and the
   model is forced to answer (budget forcing). If a trace ends within the budget, its natural
   answer counts.

Budget 0 is the run without thinking. Requires the cached runs
``run_banking77.py --think --per-class 5`` and ``run_banking77.py --per-class 5``.

Example:
    uv run python scripts/run_thinking_budget.py --models qwen3.5:2b qwen3.5:4b
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from slm import RESULTS_DIR
from slm.data import load_banking77
from slm.evaluation import LLMClassifier, stratified_subset
from slm.evaluation.thinking_budget import ThinkingBudget
from slm.inference import THINKING_OPTIONS, OllamaBackend

BUDGETS = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
OUT_DIR = RESULTS_DIR / "banking77" / "thinking_budget"


def run_cached(jobs: dict[str, Callable[[], dict]], cache_file: Path, workers: int, desc: str) -> dict[str, dict]:
    """Run the callables in ``jobs`` (key -> job), appending results to a JSONL cache keyed by ``key``."""
    done = {}
    if cache_file.exists():
        done = {r["key"]: r for r in map(json.loads, cache_file.read_text().splitlines())}
    todo = [key for key in jobs if key not in done]
    if todo:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("a") as sink, ThreadPoolExecutor(max_workers=workers) as pool:
            for key, record in tqdm(zip(todo, pool.map(lambda k: jobs[k](), todo)), total=len(todo), desc=desc):
                done[key] = record = {"key": key, **record}
                sink.write(json.dumps(record) + "\n")
                sink.flush()
    return {key: done[key] for key in jobs}


def load_run(run: str, model: str) -> pd.DataFrame:
    path = RESULTS_DIR / "banking77" / run / f"{model.replace(':', '_')}.jsonl"
    return pd.read_json(path, lines=True).set_index("idx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["qwen3.5:2b", "qwen3.5:4b"])
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--extend", type=int, default=4096, help="extra thinking tokens for truncated traces")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_banking77()
    subset = stratified_subset(dataset.test, args.per_class)
    rows = []
    for model in args.models:
        name = model.replace(":", "_")
        backend = OllamaBackend(model)
        backend.ensure_model()
        budget = ThinkingBudget(LLMClassifier(backend, dataset.labels), sampling=THINKING_OPTIONS)
        think = load_run("structured_think", model).loc[subset.index]
        no_think = load_run("structured_no-think", model).loc[subset.index]
        text = subset["text"]

        # 1. continue traces that hit the original limit
        truncated = think.index[think["done_reason"] == "length"]
        extensions = run_cached(
            {str(i): (lambda i=i: budget.extend(text[i], think.at[i, "thinking"], args.extend)) for i in truncated},
            OUT_DIR / f"{name}_extension.jsonl", args.workers, f"{model} extend",
        )
        trace = think["thinking"].copy()
        finished = think["done_reason"] == "stop"
        for key, ext in extensions.items():
            trace[int(key)] += ext["continuation"]
            finished[int(key)] = ext["finished"]
        length = trace.map(budget.count_tokens)

        # 2. natural answers of traces that finished during the extension
        late = [i for i in truncated if finished[i]]
        late_answers = run_cached(
            {str(i): (lambda i=i: budget.answer(text[i], trace[i], forced=False)) for i in late},
            OUT_DIR / f"{name}_late_answers.jsonl", args.workers, f"{model} late answers",
        )
        natural = think["prediction"].where(think["done_reason"] == "stop")
        for key, answer in late_answers.items():
            natural[int(key)] = answer["prediction"]

        # 3. forced answers wherever the budget is smaller than the natural length
        jobs = {
            f"{i}@{b}": (lambda i=i, b=b: budget.answer(text[i], budget.cut(trace[i], b), forced=True))
            for i in subset.index for b in BUDGETS
            if not (finished[i] and length[i] <= b)
        }
        forced = run_cached(jobs, OUT_DIR / f"{name}_forced.jsonl", args.workers, f"{model} forced")

        rows.append({"model": model, "budget": 0, "accuracy": (no_think["prediction"] == subset["label"]).mean(),
                     "mean_thinking_tokens": 0.0, "forced_rate": 0.0})
        for b in BUDGETS:
            within = finished & (length <= b)
            prediction = natural.where(within, [forced.get(f"{i}@{b}", {}).get("prediction") for i in subset.index])
            rows.append({
                "model": model,
                "budget": b,
                "accuracy": (prediction == subset["label"]).mean(),
                "mean_thinking_tokens": length.clip(upper=b).mean(),
                "forced_rate": 1 - within.mean(),
            })
        print(pd.DataFrame(rows[-len(BUDGETS) - 1:]).round(3).to_string(index=False))
        backend.unload()

    summary = pd.DataFrame(rows)
    summary_file = OUT_DIR / f"summary_{args.per_class}pc.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\nSaved to {summary_file}")


if __name__ == "__main__":
    main()
