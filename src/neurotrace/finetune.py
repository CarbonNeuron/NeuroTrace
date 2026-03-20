"""LoRA fine-tuning for targeted MLP layer correction."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import torch


@dataclass
class TrainingExample:
    """A single training example: prompt + expected completion token."""

    prompt: str
    completion: str
    weight: int = 1  # repetition count for weighting


@dataclass
class FinetuneConfig:
    """Configuration for a LoRA fine-tuning run."""

    target_layers: list[int]
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4
    seed: int = 42


@dataclass
class FinetuneResult:
    """Result of a fine-tuning run."""

    run_id: str
    model_name: str
    adapter_path: str
    target_layers: list[int]
    lora_rank: int
    lora_alpha: int
    dataset_name: str | None
    dataset_size: int
    epochs: int
    learning_rate: float
    seed: int | None
    train_loss_start: float | None
    train_loss_end: float | None
    scan_before: dict | None
    scan_after: dict | None
    created_at: str


def generate_training_data_from_builtin(
    dataset: list[dict],
    sabotaged_prompts: set[str] | None = None,
    sabotage_weight: int = 3,
) -> list[TrainingExample]:
    """Generate training examples from a built-in dataset.

    Args:
        dataset: List of {"prompt": ..., "answer": ...} dicts.
        sabotaged_prompts: Set of prompt strings flagged as sabotaged.
        sabotage_weight: Repetition count for sabotaged prompts.
    """
    examples = []
    for entry in dataset:
        prompt = entry["prompt"]
        answer = entry["answer"]
        # Completion must include leading space for correct tokenization
        completion = f" {answer}"
        is_sabotaged = sabotaged_prompts and prompt in sabotaged_prompts
        weight = sabotage_weight if is_sabotaged else 1
        examples.append(
            TrainingExample(
                prompt=prompt,
                completion=completion,
                weight=weight,
            )
        )
    return examples


def generate_training_data_from_jsonl(path: str) -> list[TrainingExample]:
    """Load training examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            examples.append(
                TrainingExample(
                    prompt=entry["prompt"],
                    completion=entry["completion"],
                    weight=entry.get("weight", 1),
                )
            )
    return examples


def generate_training_data_from_scan(
    scan_result,
    sabotage_weight: int = 3,
) -> list[TrainingExample]:
    """Generate training examples from scan results, weighting sabotaged prompts."""
    examples = []
    for pr in scan_result.prompt_results:
        completion = f" {pr.answer}"
        weight = sabotage_weight if pr.status == "sabotaged" else 1
        examples.append(
            TrainingExample(
                prompt=pr.prompt,
                completion=completion,
                weight=weight,
            )
        )
    return examples


def build_lora_target_modules(target_layers: list[int]) -> list[str]:
    """Build list of target module names for LoRA config."""
    modules = []
    for layer_idx in target_layers:
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            modules.append(f"model.layers.{layer_idx}.mlp.{proj}")
    return modules


def build_lora_config(config: FinetuneConfig):
    """Build a PEFT LoraConfig targeting specified MLP layers."""
    from peft import LoraConfig, TaskType

    target_modules = build_lora_target_modules(config.target_layers)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=target_modules,
        bias="none",
    )


def _expand_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Expand examples by their weight (repetition count)."""
    expanded = []
    for ex in examples:
        for _ in range(ex.weight):
            expanded.append(ex)
    return expanded


def run_finetune(
    model_name: str,
    examples: list[TrainingExample],
    config: FinetuneConfig,
    output_dir: str,
    progress_callback=None,
) -> FinetuneResult:
    """Run LoRA fine-tuning on the specified model.

    Args:
        model_name: HuggingFace model name.
        examples: Training examples.
        config: Fine-tuning configuration.
        output_dir: Directory to save adapter weights.
        progress_callback: Optional callable(description) for status updates.
    """
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    if progress_callback:
        progress_callback("Loading model for training...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        token=False,
    )

    # Apply LoRA
    if progress_callback:
        progress_callback("Applying LoRA adapter...")

    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)

    if progress_callback:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        progress_callback(
            f"Trainable: {trainable:,} / {total:,} params "
            f"({100 * trainable / total:.2f}%)"
        )

    # Expand examples by weight
    expanded = _expand_examples(examples)

    # Tokenize examples
    if progress_callback:
        progress_callback("Tokenizing training data...")

    train_encodings = _tokenize_examples(tokenizer, expanded)

    # Training loop
    if progress_callback:
        progress_callback("Starting training...")

    loss_start, loss_end = _training_loop(
        model,
        train_encodings,
        config,
        progress_callback,
    )

    # Save adapter
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if progress_callback:
        progress_callback(f"Adapter saved to {output_dir}")

    return FinetuneResult(
        run_id=run_id,
        model_name=model_name,
        adapter_path=output_dir,
        target_layers=config.target_layers,
        lora_rank=config.rank,
        lora_alpha=config.alpha,
        dataset_name=None,
        dataset_size=len(expanded),
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        train_loss_start=loss_start,
        train_loss_end=loss_end,
        scan_before=None,
        scan_after=None,
        created_at=created_at,
    )


def _tokenize_examples(tokenizer, examples: list[TrainingExample]) -> list[dict]:
    """Tokenize training examples with completion-only loss masking."""
    encodings = []
    for ex in examples:
        # Tokenize prompt and full text separately to find completion boundary
        prompt_ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
        full_text = ex.prompt + ex.completion
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Labels: -100 for prompt tokens, actual IDs for completion tokens
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        # Pad labels to match full_ids length
        if len(labels) < len(full_ids):
            labels = [-100] * (len(full_ids) - len(labels)) + labels
        elif len(labels) > len(full_ids):
            labels = labels[: len(full_ids)]

        encodings.append(
            {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
            }
        )
    return encodings


class _SimpleDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset from tokenized examples."""

    def __init__(self, encodings: list[dict]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {
            k: torch.tensor(v, dtype=torch.long) for k, v in self.encodings[idx].items()
        }


def _collate_fn(batch: list[dict]) -> dict:
    """Collate batch with left-padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    padded = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        padded["input_ids"].append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), item["input_ids"]])
        )
        padded["labels"].append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), item["labels"]])
        )
        padded["attention_mask"].append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), item["attention_mask"]])
        )
    return {k: torch.stack(v) for k, v in padded.items()}


def _training_loop(
    model,
    train_encodings: list[dict],
    config: FinetuneConfig,
    progress_callback=None,
) -> tuple[float | None, float | None]:
    """Simple training loop using AdamW with cosine schedule.

    Returns (loss_start, loss_end).
    """
    import random

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    dataset = _SimpleDataset(train_encodings)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    total_steps = len(dataloader) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    model.train()
    loss_start = None
    loss_end = None
    step = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_steps += 1

            if loss_start is None:
                loss_start = batch_loss

            loss_end = batch_loss

        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        if progress_callback:
            progress_callback(
                f"Epoch {epoch + 1}/{config.epochs} — loss: {avg_loss:.4f}"
            )

    model.eval()
    return loss_start, loss_end


def load_adapter(model, adapter_path: str):
    """Load a LoRA adapter and merge it into the model for inference.

    Args:
        model: Base model loaded via AutoModelForCausalLM.
        adapter_path: Path to saved adapter directory.

    Returns:
        Merged model (no PEFT wrapper overhead).
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model
