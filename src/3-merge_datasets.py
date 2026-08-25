#!/usr/bin/env python3
"""Merge four per-pair JSON datasets into a single combined JSON file."""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCE_FILES = [
    DATA_DIR / "presence_corresponding_object.json",
    DATA_DIR / "presence_actual_object.json",
    DATA_DIR / "false_premise_spatial.json",
    DATA_DIR / "spatial_with_llm_absent_object.json",
]

OUTPUT_FILE = DATA_DIR / "merged_dataset.json"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge(sources: list[Path], output: Path):
    datasets = []
    for src in sources:
        data = load_json(src)
        datasets.append(data)
        print(f"Loaded {src.name}: {len(data['items'])} items")

    n = len(datasets[0]["items"])
    for ds in datasets[1:]:
        assert len(ds["items"]) == n, (
            f"Item count mismatch: expected {n}, got {len(ds['items'])}"
        )

    merged_items = []
    for i in range(n):
        items = [ds["items"][i] for ds in datasets]

        common_keys = {"source_image", "原物体", "对应物体"}
        for key in common_keys:
            vals = {it.get(key) for it in items}
            assert len(vals) == 1, (
                f"Row {i}: key '{key}' differs across files: {vals}"
            )

        base = {k: items[0][k] for k in common_keys}

        for it in items:
            kind = it.get("dataset_kind") or it.get("relation_type", "unknown")
            prefix = it.get("dataset_kind", it.get("relation_type", ""))
            if "dataset_kind" in it:
                prefix = it["dataset_kind"]
            else:
                prefix = "false_premise_spatial"

            entry = {k: v for k, v in it.items() if k not in common_keys}
            base[prefix] = entry

        merged_items.append(base)

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": [str(s) for s in sources],
            "item_count": len(merged_items),
        },
        "items": merged_items,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nMerged {len(merged_items)} items -> {output}")


def main():
    parser = argparse.ArgumentParser(description="Merge four FP-POPE datasets")
    parser.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_FILE,
        help="Output path (default: data/merged_dataset.json)",
    )
    args = parser.parse_args()
    merge(SOURCE_FILES, args.output)


if __name__ == "__main__":
    main()
