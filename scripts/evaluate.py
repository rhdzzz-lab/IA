from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned BERT classifier.")
    parser.add_argument("--model-dir", required=True, help="Path to a saved model directory.")
    parser.add_argument("--data-file", required=True, help="Path to a CSV or JSONL file.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", help="Optional JSON file to store the metrics.")
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def load_label_names(model, metadata_path: Path) -> List[str]:
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = metadata.get("labels")
            if isinstance(labels, list) and labels:
                return [str(label) for label in labels]
        except json.JSONDecodeError:
            pass

    id2label = getattr(model.config, "id2label", {}) or {}
    label_names: List[str] = []
    for idx in range(int(model.config.num_labels)):
        label = id2label.get(idx)
        if label is None:
            label = id2label.get(str(idx), f"LABEL_{idx}")
        label_names.append(str(label))
    return label_names


def batched(items: Sequence[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def predict(model, tokenizer, texts: Sequence[str], device: torch.device, max_length: int, batch_size: int):
    predictions: List[int] = []
    probabilities: List[np.ndarray] = []

    model.eval()
    for batch_texts in batched(texts, batch_size):
        encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        predictions.extend(torch.argmax(probs, dim=-1).cpu().tolist())
        probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities


def main() -> None:
    args = parse_args()
    device = select_device()
    model_dir = Path(args.model_dir)

    df = load_dataframe(Path(args.data_file))
    _, eval_df = split_dataframe(df, args.test_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    label_names = load_label_names(model, model_dir / "training_metadata.json")

    y_true = eval_df["label"].tolist()
    y_pred_ids, _ = predict(model, tokenizer, eval_df["text"].tolist(), device, args.max_length, args.batch_size)
    y_true_ids = [label_names.index(label) for label in y_true]
    y_pred = [label_names[idx] for idx in y_pred_ids]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_names).tolist()
    report = classification_report(y_true, y_pred, labels=label_names, output_dict=True, zero_division=0)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "labels": label_names,
        "classification_report": report,
    }

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
