# small-language-models
Repository for a small language model talk: hands-on examples of what small language models can (and cannot) do.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com) server.

```bash
uv sync                                    # creates .venv with all dependencies
uv run jupyter lab                         # notebooks live in notebooks/
```

Fine-tuning (notebook 07) additionally needs PyTorch and Unsloth, in the optional dependency group `finetune`
(a few GB; needs an NVIDIA GPU):

```bash
uv sync --group finetune
uv run --group finetune jupyter lab
```

## Layout

```
src/slm/
  data/          dataset loaders (downloaded on first use into data/)
  inference/     inference backends (Ollama)
  evaluation/    task-specific evaluation (LLM classifier, metrics)
  finetuning.py  LoRA fine-tuning of a small model with Unsloth
scripts/         command-line experiment runners
notebooks/       walkthroughs and result analysis
results/         experiment outputs (per-example predictions are git-ignored)
```

## Experiments

### 1. BANKING77 intent classification vs. model size

Zero-shot classification of the BANKING77 test set (3,080 queries, 77 intents) with Qwen models of
increasing size. The label set is enforced via Ollama structured outputs (JSON schema `enum`);
thinking is off and decoding is greedy.

* Notebook: `notebooks/01_banking77_qwen_scaling.ipynb`
* Script: `uv run python scripts/run_banking77.py` (Qwen3.5 0.8b, 2b, 4b, 9b on the full test set;
  `--per-class 5` for a quick balanced subset, `--models ...` for other ladders)

Predictions are cached per model in `results/banking77/<mode>_<think|no-think>/`, so interrupted runs resume.
Latencies are measured with `--workers` concurrent requests (default 4); set
`OLLAMA_NUM_PARALLEL` on the server accordingly.

### 2. Output control and reasoning

* **Structured vs. open answers**: constrained decoding (JSON schema `enum`) vs. free text, scored with strict
  and lenient label matching; Qwen3.5 0.8b-9b and qwen3.6:35b-a3b.
* **Explanation in the JSON answer**: a short `reason` before or after the (enum-constrained) `label`.
* **Reasoning**: thinking mode on vs. off for a small and a medium model.
* **Thinking budget**: budget forcing (cut the reasoning after 256 ... 8k tokens and force an answer) and
  extension of traces that hit the 4k limit, giving accuracy as a function of the thinking budget.
* Notebook: `notebooks/02_banking77_output_control_and_reasoning.ipynb`
* Scripts: `scripts/run_banking77.py --mode {structured,json,free} [--think] [--explanation {before,after}]`,
  `scripts/run_thinking_budget.py` (needs the cached `--think --per-class 5` runs)

### 3. Prompt optimization with DSPy

The same task as a DSPy program: zero-shot, chain of thought, random few-shot and MIPROv2 (a larger model
proposes instructions for the small task model).

* Notebook: `notebooks/03_banking77_dspy.ipynb`
* Script: `uv run python scripts/run_dspy_banking77.py --task-model qwen3.5:2b --prompt-model qwen3.5:9b`
* Other models (zero-shot and few-shot only): `uv run python scripts/run_dspy_banking77.py --task-model qwen3.5:4b --prompt-model qwen3.5:4b --programs zero-shot few-shot`

### 4. Calibration: does a small model know when it is wrong?

Three confidence signals: the confidence the model states in its answer, the probability of the generated
label (token log-probabilities, renormalized over the labels allowed by the schema) and the agreement of
10 sampled answers. Compared with reliability diagrams, ECE and AUROC, and used to route uncertain queries
of `qwen3.5:4b` to `qwen3.6:35b-a3b` (cascade).

* Notebook: `notebooks/04_banking77_calibration.ipynb`
* Script: `uv run python scripts/run_calibration.py --signal {label_prob,stated,sampling}`

### 5. Reasoning effort with gpt-oss

`gpt-oss:20b` (21B parameters, 3.6B active) with reasoning effort low, medium and high, compared with the
Qwen ladder.

* Notebook: `notebooks/05_banking77_reasoning_effort.ipynb`
* Script: `uv run python scripts/run_banking77.py --models gpt-oss:20b --effort {low,medium,high} --per-class 5 --logprobs`

### 6. Batch prompting

Several numbered queries per call (1 to 40), with one constrained label per query, for `qwen3.5:2b`, `4b`, `9b` and `qwen3.6:35b-a3b`: accuracy vs. batch size,
by position in the batch, spill-over between neighbouring queries, and the saving in prompt tokens and time.

* Notebook: `notebooks/06_banking77_batching.ipynb`
* Script: `uv run python scripts/run_batching.py --models qwen3.5:2b qwen3.5:4b qwen3.5:9b qwen3.6:35b-a3b --batch-sizes 1 5 10 20 40 --seeds 0 1`

### 7. Fine-tuning a 0.8B model with Unsloth

LoRA fine-tuning of `Qwen3.5-0.8B` on 1 to all (about 130) BANKING77 training queries per intent, compared with
the prompted models. The fine-tuned model no longer needs the label list in the prompt.

* Notebook: `notebooks/07_banking77_finetuning.ipynb`
* Script: `uv run --group finetune python scripts/finetune_banking77.py --train-per-class 0 1 5 10 25 50 all --epochs 3 --save-adapter`
