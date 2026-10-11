"""Optimize a BANKING77 intent classifier for a small model with DSPy.

Compares, for one (small) task model:
    zero-shot   dspy.Predict with DSPy's default prompt
    cot         dspy.ChainOfThought: the model writes a short rationale before answering
    few-shot    16 random labelled training examples in the prompt (LabeledFewShot)
    mipro       MIPROv2: a larger "prompt model" proposes instructions, and instructions plus
                few-shot examples are searched on a validation set

All programs are evaluated on the same balanced test subset (default: 5 queries per intent).

Examples:
    uv run python scripts/run_dspy_banking77.py
    uv run python scripts/run_dspy_banking77.py --task-model qwen3.5:4b --programs zero-shot mipro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dspy
import pandas as pd

from slm import RESULTS_DIR
from slm.data import load_banking77
from slm.evaluation import stratified_subset
from slm.evaluation.dspy_classification import (
    evaluate_program,
    exact_match,
    intent_signature,
    ollama_lm,
    to_examples,
)
from slm.inference import OllamaBackend

PROGRAMS = ("zero-shot", "cot", "few-shot", "mipro")
OUT_DIR = RESULTS_DIR / "banking77" / "dspy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-model", default="qwen3.5:2b", help="the small model whose prompt is optimized")
    parser.add_argument("--prompt-model", default="qwen3.5:9b", help="model that proposes instructions (MIPROv2)")
    parser.add_argument("--programs", nargs="+", choices=PROGRAMS, default=list(PROGRAMS))
    parser.add_argument("--test-per-class", type=int, default=5)
    parser.add_argument("--train-per-class", type=int, default=2, help="training examples per intent")
    parser.add_argument("--val-per-class", type=int, default=2, help="validation examples per intent (MIPROv2)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_banking77()
    testset = to_examples(stratified_subset(dataset.test, args.test_per_class, seed=args.seed))
    train_pool = stratified_subset(dataset.train, args.train_per_class + args.val_per_class, seed=args.seed)
    trainset = to_examples(train_pool.groupby("label", group_keys=False).head(args.train_per_class))
    valset = to_examples(train_pool.drop(index=[ex.idx for ex in trainset]))

    for model in {args.task_model, args.prompt_model}:
        OllamaBackend(model).ensure_model()
    task_lm = ollama_lm(args.task_model)
    dspy.configure(lm=task_lm)
    signature = intent_signature(dataset.labels)
    run_dir = OUT_DIR / args.task_model.replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in args.programs:
        print(f"\n=== {name} ===")
        if name == "zero-shot":
            program = dspy.Predict(signature)
        elif name == "cot":
            program = dspy.ChainOfThought(signature)
        elif name == "few-shot":
            program = dspy.LabeledFewShot(k=16).compile(dspy.Predict(signature), trainset=trainset)
        else:  # mipro
            optimizer = dspy.MIPROv2(
                metric=exact_match,
                prompt_model=ollama_lm(args.prompt_model, temperature=1.0),
                task_model=task_lm,
                auto="light",
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
                num_threads=args.threads,
                seed=args.seed,
            )
            program = optimizer.compile(dspy.Predict(signature), trainset=trainset, valset=valset)

        program.save(run_dir / f"{name}.json")
        metrics = evaluate_program(program, testset, args.threads, output_file=run_dir / f"{name}_predictions.jsonl")
        summary = update_summary(run_dir / "summary.csv", {"task_model": args.task_model, "program": name, **metrics})
        print(f"{name}: accuracy={metrics['accuracy']:.3f}")

    print(f"\n{summary.to_string(index=False)}\n\nSaved to {run_dir / 'summary.csv'}")


def update_summary(summary_file: Path, row: dict) -> pd.DataFrame:
    """Add or replace one program's row, saving after every program so a crash keeps finished results."""
    summary = pd.DataFrame([row])
    if summary_file.exists():
        previous = pd.read_csv(summary_file)
        summary = pd.concat([previous[previous["program"] != row["program"]], summary], ignore_index=True)
    summary.to_csv(summary_file, index=False)
    return summary


if __name__ == "__main__":
    main()
