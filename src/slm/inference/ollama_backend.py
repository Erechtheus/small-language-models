"""Thin wrapper around the Ollama Python client."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import ollama


def params_in_billions(model: str) -> float | None:
    """Parse the (total) parameter count from an Ollama tag such as ``qwen3.5:0.8b`` or ``qwen3.6:35b-a3b``."""
    match = re.search(r":(\d+(?:\.\d+)?)b", model)
    return float(match.group(1)) if match else None


# Qwen's recommended sampling for thinking mode (greedy decoding tends to loop), plus room for the
# reasoning trace and a cap so that a runaway trace cannot block a run.
THINKING_OPTIONS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_ctx": 8192, "num_predict": 4096}

# gpt-oss with a reasoning effort: OpenAI's recommended sampling, and room for long traces at "high"
EFFORT_OPTIONS = {"temperature": 1.0, "top_p": 1.0, "num_ctx": 10240, "num_predict": 8192}

# gpt-oss always reasons; with ``think="none"`` the prompt is written in its chat format (harmony) by hand,
# with an empty reasoning message, so that the model has to start with the final answer right away
HARMONY_SYSTEM = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\nCurrent date: {today}\n\nReasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
)
HARMONY_NO_REASONING = "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"


def harmony_prompt_without_reasoning(messages: list[dict[str, str]]) -> str:
    """The prompt Ollama's gpt-oss template would build, followed by an empty reasoning message."""
    prompt = HARMONY_SYSTEM.format(today=date.today().isoformat())
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    if system:
        prompt += f"<|start|>developer<|message|># Instructions\n\n{system}<|end|>"
    for m in messages:
        if m["role"] == "user":
            prompt += f"<|start|>user<|message|>{m['content']}<|end|>"
        elif m["role"] == "assistant":
            prompt += f"<|start|>assistant<|channel|>final<|message|>{m['content']}<|end|>"
    return prompt + HARMONY_NO_REASONING


@dataclass
class Completion:
    content: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int  # includes reasoning tokens when thinking is enabled
    thinking: str = ""
    done_reason: str = ""  # "length" means the output hit num_predict and was truncated
    # per generated token: (token, logprob, [(alternative, logprob), ...]); only with ``top_logprobs``
    logprobs: list[tuple[str, float, list[tuple[str, float]]]] | None = None


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        think: bool | str = False,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        keep_alive: str = "10m",
        top_logprobs: int = 0,
        **options: Any,
    ):
        """``think`` can also be a reasoning effort (``"low"``, ``"medium"``, ``"high"``) for models
        that support it, such as gpt-oss, or ``"none"`` to skip gpt-oss's reasoning (see
        ``harmony_prompt_without_reasoning``). ``top_logprobs`` (at most 20) returns the probabilities of
        the generated tokens and their most likely alternatives.
        ``options`` are passed on to Ollama, e.g. ``num_predict``, ``top_p`` or ``top_k``."""
        self.model = model
        self.client = ollama.Client(host=host)
        self.think = think
        self.top_logprobs = top_logprobs
        self.options = {"temperature": temperature, "num_ctx": num_ctx, **options}
        self.keep_alive = keep_alive

    def ensure_model(self, pull: bool = True) -> None:
        """Make sure the model is available locally (pulling it if requested) and loaded into memory.

        Loading up front keeps the model's cold start out of the measured latencies.
        """
        local = {m.model for m in self.client.list().models}
        if self.model not in local:
            if not pull:
                raise RuntimeError(f"Model {self.model!r} not available locally (ollama pull {self.model})")
            self.client.pull(self.model)
        # same num_ctx as the later calls: a different context size makes Ollama reload the model
        self.client.generate(model=self.model, prompt="", keep_alive=self.keep_alive,
                             options={"num_ctx": self.options["num_ctx"]})

    def chat(self, messages: list[dict[str, str]], format: dict[str, Any] | str | None = None) -> Completion:
        if self.think == "none":
            return self._generate_without_reasoning(messages, format)
        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=format,
            think=self.think,
            options=self.options,
            keep_alive=self.keep_alive,
            logprobs=self.top_logprobs > 0,
            top_logprobs=self.top_logprobs or None,
        )
        return Completion(
            content=response.message.content or "",
            latency_s=time.perf_counter() - start,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            thinking=response.message.thinking or "",
            done_reason=response.done_reason or "",
            logprobs=self._logprobs(response),
        )

    @staticmethod
    def _logprobs(response: Any) -> list[tuple[str, float, list[tuple[str, float]]]] | None:
        if not response.logprobs:
            return None
        return [(t.token, t.logprob, [(a.token, a.logprob) for a in t.top_logprobs or []])
                for t in response.logprobs]

    def _generate_without_reasoning(self, messages: list[dict[str, str]],
                                    format: dict[str, Any] | str | None = None) -> Completion:
        start = time.perf_counter()
        response = self.client.generate(
            model=self.model,
            prompt=harmony_prompt_without_reasoning(messages),
            raw=True,
            format=format,
            options=self.options,
            keep_alive=self.keep_alive,
            logprobs=self.top_logprobs > 0,
            top_logprobs=self.top_logprobs or None,
        )
        return Completion(
            content=response.response or "",
            latency_s=time.perf_counter() - start,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            done_reason=response.done_reason or "",
            logprobs=self._logprobs(response),
        )

    def unload(self) -> None:
        """Free GPU memory before switching to the next model."""
        self.client.generate(model=self.model, prompt="", keep_alive=0)
