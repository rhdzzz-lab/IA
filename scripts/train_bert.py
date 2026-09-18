from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT classifier from CSV or JSONL data."
    )
    parser.add_argument("--data-file", required=True, help="Path to a CSV or JSONL file.")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/bert_classifier")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataframe(data_file: Path) -> pd.DataFrame:
    suffix = data_file.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_file)
    elif suffix == ".jsonl":
        df = pd.read_json(data_file, lines=True)
    else:
        raise ValueError("The input file must be .csv or .jsonl")

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("The file must contain the columns 'label' and 'text'.")

    df = df[["label", "text"]].dropna().copy()
    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["label"] != "") & (df["text"] != "")]
    return df.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_counts = df["label"].value_counts()
    n_labels = label_counts.shape[0]
    n_eval = max(1, math.ceil(len(df) * test_size))

    can_stratify = n_labels > 1 and int(label_counts.min()) >= 2 and n_eval >= n_labels
    stratify = df["label"] if can_stratify else None

    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_training_arguments(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: Dict[str, object] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "seed": args.seed,
    }

    if "report_to" in params:
        kwargs["report_to"] = "none"
    if "logging_steps" in params:
        kwargs["logging_steps"] = 10
    if "save_total_limit" in params:
        kwargs["save_total_limit"] = 2
    if "load_best_model_at_end" in params:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:
        kwargs["metric_for_best_model"] = "f1"
    if "greater_is_better" in params:
        kwargs["greater_is_better"] = True
    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in params:
        kwargs["logging_strategy"] = "steps"
    if "remove_unused_columns" in params:
        kwargs["remove_unused_columns"] = False
    if "disable_tqdm" in params:
        kwargs["disable_tqdm"] = False
    if "use_mps_device" in params and torch.backends.mps.is_available():
        kwargs["use_mps_device"] = True

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def remove_text_column(dataset: Dataset) -> Dataset:
    if "text" in dataset.column_names:
        return dataset.remove_columns(["text"])
    return dataset


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(data_file)
    labels = sorted(df["label"].unique().tolist())
    label2id: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}
    id2label: Dict[int, str] = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    train_df, eval_df = split_dataframe(df, args.test_size, args.seed)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )
    eval_dataset = Dataset.from_pandas(
        eval_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_dataset = remove_text_column(train_dataset.map(tokenize, batched=True))
    eval_dataset = remove_text_column(eval_dataset.map(tokenize, batched=True))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_arguments(args, output_dir)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "data_file": str(data_file),
        "model_name": args.model_name,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    }

    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Model saved to {output_dir}")
    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
