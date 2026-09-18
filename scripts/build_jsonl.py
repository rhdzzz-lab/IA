from __future__ import annotations

import csv
import json
from pathlib import Path


def convert_csv_to_jsonl(input_dir: Path, output_dir: Path) -> None:
    """Convert every CSV file in input_dir to a JSONL file in output_dir.

    Expected CSV columns:
    - label
    - text
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for csv_path in csv_files:
        jsonl_path = output_dir / f"{csv_path.stem}.jsonl"
        with csv_path.open("r", encoding="utf-8", newline="") as f_in, jsonl_path.open(
            "w", encoding="utf-8"
        ) as f_out:
            reader = csv.DictReader(f_in)
            if not reader.fieldnames or "label" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(
                    f"{csv_path.name} must contain the columns 'label' and 'text'."
                )

            for row in reader:
                label = (row.get("label") or "").strip()
                text = (row.get("text") or "").strip()
                if not label or not text:
                    continue
                f_out.write(json.dumps({"label": label, "text": text}, ensure_ascii=False) + "\n")

        print(f"Created {jsonl_path.relative_to(output_dir.parent)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = repo_root / "datasets"
    output_dir = repo_root / "jsonl"
    convert_csv_to_jsonl(input_dir, output_dir)


if __name__ == "__main__":
    main()
