from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify new texts with a fine-tuned BERT model.")
    parser.add_argument("--model-dir", required=True, help="Path to a saved model directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text to classify.")
    group.add_argument("--text-file", help="Path to a UTF-8 file with one text per line.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def classify_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    label_names: Sequence[str],
    device: torch.device,
    max_length: int,
    top_k: int,
):
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)

    top_k = max(1, min(top_k, probabilities.shape[-1]))
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

    results = []
    for text, probs, indices in zip(texts, top_probs, top_indices):
        predictions = [
            {"label": label_names[int(index)], "score": float(score)}
            for score, index in zip(probs.tolist(), indices.tolist())
        ]
        results.append(
            {
                "text": text,
                "top_prediction": predictions[0],
                "predictions": predictions,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    device = select_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    label_names = load_label_names(model, model_dir / "training_metadata.json")

    if args.text_file:
        texts = [line.strip() for line in Path(args.text_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        texts = [args.text.strip()]

    if not texts:
        raise ValueError("No text to classify.")

    results = classify_texts(texts, tokenizer, model, label_names, device, args.max_length, args.top_k)
    print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
