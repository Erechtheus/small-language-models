"""Name audit (notebook 00, section 3c): profession guesses and hiring scores per first name.

For every model, all prompts from ``slm.evaluation.name_bias.audit_prompts`` are answered; the probability
of every allowed answer is stored. One CSV per model; existing files are skipped.

Example:
    uv run python scripts/run_name_bias.py --models qwen3.5:0.8b qwen3.5:2b qwen3.5:4b qwen3.5:9b qwen3.6:35b-a3b
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import ollama
import pandas as pd
from tqdm import tqdm

from slm import RESULTS_DIR
from slm.evaluation.name_bias import audit_prompts, choice_distribution

OUT_DIR = RESULTS_DIR / "bias" / "names"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.6:35b-a3b"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    client = ollama.Client()
    prompts = audit_prompts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        out = OUT_DIR / f"{model.replace(':', '_')}.csv"
        if out.exists():
            print(f"{model}: {out} exists, skipped")
            continue
        client.generate(model=model, prompt="", keep_alive="5m", options={"num_ctx": 1024})  # load once
        with ThreadPoolExecutor(args.workers) as pool:
            dists = list(tqdm(pool.map(lambda p: choice_distribution(client, model, p["prompt"], p["options"]),
                                       prompts), total=len(prompts), desc=model))
        rows = [{**{k: v for k, v in p.items() if k != "options"}, "model": model,
                 "probs": json.dumps(d) if d else None} for p, d in zip(prompts, dists)]
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{model}: {sum(d is None for d in dists)} of {len(dists)} answers without a distribution")
        client.generate(model=model, prompt="", keep_alive=0)


if __name__ == "__main__":
    main()
