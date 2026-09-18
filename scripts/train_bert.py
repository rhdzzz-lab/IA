from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Fine-tune a BERT classifier from CSV or JSONL data.")
    parser.add_argument("--data-file", required=True, help="Path to a CSV or JSONL file.")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/bert_classifier")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
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
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
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

    train_df, eval_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label_id"] if len(labels) > 1 else None,
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "data_file": str(data_file),
        "model_name": args.model_name,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
