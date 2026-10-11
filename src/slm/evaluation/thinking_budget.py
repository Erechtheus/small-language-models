"""Thinking budgets for Qwen models: budget forcing and extension of reasoning traces.

Works on reasoning traces that were already generated (``--think`` runs of ``run_banking77.py``):

* **Budget forcing**: cut a trace after ``n`` tokens, close the thinking block and make the
  model answer immediately (as in s1, Muennighoff et al. 2025, and Qwen3's thinking budget).
* **Extension**: continue a trace that hit the token limit, to see whether more room helps.

Both need control over the raw prompt, so we bypass Ollama's chat template and send Qwen's
template ourselves (``raw=True``). Token positions are exact thanks to the original tokenizer.
"""

from __future__ import annotations

import time
from functools import cached_property

import ollama
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from slm.evaluation.classification import LLMClassifier

# Qwen3.5 chat template with thinking enabled, up to the start of the reasoning trace
QWEN_THINKING_PROMPT = (
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
END_THINKING = "\n</think>\n\n"
FORCE_ANSWER = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now."
    + END_THINKING
)

# Ollama tag -> Hugging Face repository with the matching tokenizer
TOKENIZER_REPOS = {"qwen3.5:0.8b": "Qwen/Qwen3.5-0.8B", "qwen3.5:2b": "Qwen/Qwen3.5-2B",
                   "qwen3.5:4b": "Qwen/Qwen3.5-4B", "qwen3.5:9b": "Qwen/Qwen3.5-9B"}


class ThinkingBudget:
    def __init__(self, classifier: LLMClassifier, sampling: dict, num_ctx: int = 10240):
        """``classifier`` provides model, system prompt and answer schema (use ``mode="structured"``);
        ``sampling`` are the options used for continuing traces (same as for generating them)."""
        self.classifier = classifier
        self.model = classifier.backend.model
        self.client = ollama.Client()
        self.sampling = {**sampling, "num_ctx": num_ctx}
        self.num_ctx = num_ctx

    @cached_property
    def tokenizer(self) -> Tokenizer:
        return Tokenizer.from_file(hf_hub_download(TOKENIZER_REPOS[self.model], "tokenizer.json"))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def cut(self, trace: str, n_tokens: int) -> str:
        """The first ``n_tokens`` tokens of a reasoning trace."""
        ids = self.tokenizer.encode(trace, add_special_tokens=False).ids
        return self.tokenizer.decode(ids[:n_tokens], skip_special_tokens=False)

    def prompt(self, text: str) -> str:
        return QWEN_THINKING_PROMPT.format(system=self.classifier.system_prompt, user=text)

    def extend(self, text: str, trace: str, max_new_tokens: int) -> dict:
        """Let the model keep thinking after ``trace`` for at most ``max_new_tokens`` tokens."""
        start = time.perf_counter()
        response = self.client.generate(
            model=self.model, prompt=self.prompt(text) + trace, raw=True,
            options={**self.sampling, "num_predict": max_new_tokens, "stop": ["</think>"]},
        )
        return {
            "continuation": response.response,
            "continuation_tokens": response.eval_count or 0,
            "finished": response.done_reason == "stop",  # reached </think> instead of the token limit
            "latency_s": time.perf_counter() - start,
        }

    def answer(self, text: str, trace: str, forced: bool) -> dict:
        """Close the thinking block after ``trace`` and let the model answer (constrained, greedy).

        With ``forced=True`` the model is told that it has to answer now (budget forcing);
        otherwise the trace ended naturally and is just closed.
        """
        start = time.perf_counter()
        response = self.client.generate(
            model=self.model, prompt=self.prompt(text) + trace + (FORCE_ANSWER if forced else END_THINKING),
            raw=True, format=self.classifier.format,
            options={"temperature": 0.0, "num_ctx": self.num_ctx, "num_predict": 50},
        )
        strict, _ = self.classifier.parse(response.response)
        return {"prediction": strict, "raw": response.response, "latency_s": time.perf_counter() - start}
