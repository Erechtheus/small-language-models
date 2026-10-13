"""Fine-tuning a small model as an intent classifier with Unsloth (LoRA).

Needs the optional ``finetune`` dependency group: ``uv run --group finetune ...``.
The model learns to answer with the label directly, so the prompt no longer needs the list of 77 intents.
"""

from __future__ import annotations

import time

import pandas as pd

from slm.evaluation.classification import INVALID

FINETUNE_SYSTEM_PROMPT = "Classify the intent of the customer message sent to a bank. Answer with the intent label only."


def build_prompt(tokenizer, text: str) -> str:
    """The chat-formatted prompt up to the start of the answer (thinking switched off)."""
    messages = [{"role": "system", "content": FINETUNE_SYSTEM_PROMPT}, {"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_model(model_name: str, max_seq_length: int = 256, lora_rank: int | None = 16, seed: int = 0):
    """Load the model in 16 bit; with ``lora_rank`` it gets trainable LoRA adapters on all linear layers."""
    from unsloth import FastLanguageModel

    model, processor = FastLanguageModel.from_pretrained(model_name, max_seq_length=max_seq_length, load_in_4bit=False)
    # Qwen3.5 is a vision-language model; its processor wraps the text tokenizer
    tokenizer = getattr(processor, "tokenizer", processor)
    if lora_rank:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
    return model, tokenizer


def train(model, tokenizer, data: pd.DataFrame, output_dir: str, epochs: float = 1.0, learning_rate: float = 2e-4,
          batch_size: int = 16, seed: int = 0) -> dict:
    """Supervised fine-tuning on (prompt, label) pairs; the loss is computed on the label tokens only."""
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    # Tokenize here (much faster than the trainer's own preparation); prompt tokens get the label -100,
    # so only the answer is learned
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for text, label in zip(data["text"], data["label"]):
        prompt = tokenizer(build_prompt(tokenizer, text), add_special_tokens=False)["input_ids"]
        answer = tokenizer(f"{label}{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]
        rows["input_ids"].append(prompt + answer)
        rows["attention_mask"].append([1] * (len(prompt) + len(answer)))
        rows["labels"].append([-100] * len(prompt) + answer)
    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict(rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            lr_scheduler_type="linear",
            warmup_steps=10,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=seed,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
        ),
    )
    stats = trainer.train()
    return {"train_loss": stats.training_loss, "train_time_s": stats.metrics["train_runtime"],
            "steps": stats.global_step, "log": trainer.state.log_history}


def predict(model, tokenizer, texts: list[str], labels: list[str], batch_size: int = 64,
            max_new_tokens: int = 24) -> tuple[list[str], list[str], float]:
    """Greedy answers for ``texts``; returns (predictions, raw answers, seconds per query)."""
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    raw = []
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        prompts = [build_prompt(tokenizer, text) for text in texts[i:i + batch_size]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        raw += tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    seconds = (time.perf_counter() - start) / len(texts)
    valid = set(labels)
    predictions = [answer.strip() if answer.strip() in valid else INVALID for answer in raw]
    return predictions, raw, seconds
