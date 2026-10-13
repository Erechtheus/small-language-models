"""Fine-tune a small Qwen model on BANKING77 with Unsloth (LoRA) and evaluate it.

For each training-set size (examples per intent; "all" uses the full training set of 10,003 queries), a fresh
LoRA adapter is trained and evaluated on the full test set (3,080 queries). Accuracy is also reported on the
balanced subset (5 per intent) used in the other notebooks. Size 0 evaluates the model without training.

Examples:
    uv run --group finetune python scripts/finetune_banking77.py
    uv run --group finetune python scripts/finetune_banking77.py --train-per-class 10 all --epochs 2
"""

from __future__ import annotations

import argparse
import gc
import json

import pandas as pd

from slm import RESULTS_DIR
from slm.data import load_banking77
from slm.evaluation import stratified_subset

OUT_DIR = RESULTS_DIR / "banking77" / "finetune"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--train-per-class", nargs="+", default=["0", "10", "all"],
                        help="training examples per intent; 'all' for the full training set")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-adapter", action="store_true")
    args = parser.parse_args()

    import torch

    from slm.finetuning import load_model, predict, train

    dataset = load_banking77()
    subset_idx = stratified_subset(dataset.test, per_class=5).index
    run_dir = OUT_DIR / args.model.split("/")[-1]
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run_dir / "summary.csv"

    for size in args.train_per_class:
        name = f"train-{size}"
        n = None if size == "all" else int(size)
        print(f"\n=== {name} ===")
        model, tokenizer = load_model(args.model, lora_rank=args.lora_rank if n != 0 else None, seed=args.seed)
        stats = {"train_loss": None, "train_time_s": 0.0, "steps": 0, "log": []}
        if n != 0:
            if n is None:
                data = dataset.train
            elif n <= dataset.train["label"].value_counts().min():
                data = stratified_subset(dataset.train, n, seed=args.seed)
            else:  # intents with fewer than n training examples contribute all of them
                data = pd.concat([group.sample(n=min(n, len(group)), random_state=args.seed)
                                  for _, group in dataset.train.groupby("label")])
            data = data.sample(frac=1.0, random_state=args.seed)
            stats = train(model, tokenizer, data, str(run_dir / "checkpoints"), epochs=args.epochs,
                          learning_rate=args.learning_rate, batch_size=args.batch_size, seed=args.seed)
        predictions, raw, seconds = predict(model, tokenizer, dataset.test["text"].tolist(), dataset.labels)

        results = dataset.test.assign(prediction=predictions, raw=raw)
        correct = results["prediction"] == results["label"]
        with (run_dir / f"{name}_predictions.jsonl").open("w") as f:
            for idx, row in results.iterrows():
                f.write(json.dumps({"idx": int(idx), "prediction": row.prediction, "raw": row.raw}) + "\n")
        (run_dir / f"{name}_log.json").write_text(json.dumps(stats["log"]))
        if args.save_adapter and n != 0:
            model.save_pretrained(run_dir / f"{name}_adapter")
            tokenizer.save_pretrained(run_dir / f"{name}_adapter")

        row = {
            "model": args.model, "train_per_class": size, "train_examples": 0 if n == 0 else len(data),
            "epochs": args.epochs if n != 0 else 0, "lora_rank": args.lora_rank, "learning_rate": args.learning_rate,
            "accuracy_test": correct.mean(), "accuracy_subset": correct.loc[subset_idx].mean(),
            "invalid_rate": (results["prediction"] == "<invalid>").mean(),
            "train_loss": stats["train_loss"], "train_time_s": stats["train_time_s"], "steps": stats["steps"],
            "seconds_per_query": seconds,
        }
        summary = pd.DataFrame([row])
        if summary_file.exists():
            previous = pd.read_csv(summary_file, dtype={"train_per_class": str})
            summary = pd.concat([previous[previous["train_per_class"] != size], summary], ignore_index=True)
        summary.to_csv(summary_file, index=False)
        print(pd.DataFrame([row]).drop(columns="model").round(3).to_string(index=False))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nSaved to {summary_file}")


if __name__ == "__main__":
    main()
