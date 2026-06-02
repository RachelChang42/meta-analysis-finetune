import argparse
import glob
import json
from pathlib import Path
from collections import Counter, defaultdict


FIELDS = [
    "Literature Title",
    "Study Area & Country",
    "Data Year",
    "Accessibility Method",
    "Facility Type",
    "Demand Population",
    "Dist/Time Calc Method",
    "Transport Mode",
    "Travel Time Period",
    "Urbanization Rate",
]

OPTIONAL_FIELDS = ["序号"]

EMPTY_VALUES = {"", "N/A", "NA", "None", "null", None}


def load_answer_items(line: str):
    obj = json.loads(line)
    messages = obj["messages"]
    assistant_content = messages[2]["content"]
    answer_items = json.loads(assistant_content)
    return answer_items


def is_empty(value):
    if value is None:
        return True
    return str(value).strip() in EMPTY_VALUES


def analyze_file(path: str):
    sample_count = 0
    answer_item_count = 0
    item_count_per_sample = Counter()
    empty_field_count = Counter()
    field_type_count = defaultdict(Counter)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample_count += 1
            answer_items = load_answer_items(line)

            item_count_per_sample[len(answer_items)] += 1
            answer_item_count += len(answer_items)

            for item in answer_items:
                for field in FIELDS:
                    value = item.get(field)
                    field_type_count[field][type(value).__name__] += 1

                    if is_empty(value):
                        empty_field_count[field] += 1

    return {
        "sample_count": sample_count,
        "answer_item_count": answer_item_count,
        "item_count_per_sample": item_count_per_sample,
        "empty_field_count": empty_field_count,
        "field_type_count": field_type_count,
    }


def print_file_stats(path: str, stats: dict):
    print("=" * 80)
    print(f"File: {path}")
    print(f"Samples: {stats['sample_count']}")
    print(f"Answer items: {stats['answer_item_count']}")

    print("\nAnswer items per sample:")
    for item_num, count in sorted(stats["item_count_per_sample"].items()):
        print(f"  {item_num} item(s): {count} sample(s)")

    print("\nEmpty/N/A fields:")
    for field in FIELDS:
        empty_count = stats["empty_field_count"][field]
        print(f"  {field}: {empty_count}")

    print("\nField value types:")
    for field in FIELDS:
        types = dict(stats["field_type_count"][field])
        print(f"  {field}: {types}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/folds",
        help="Directory containing train_fold_*.jsonl and val_fold_*.jsonl",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        raise SystemExit(1)

    paths = sorted(glob.glob(str(data_dir / "*.jsonl")))

    if not paths:
        print(f"No .jsonl files found in: {data_dir}")
        raise SystemExit(1)

    total_samples = 0
    total_answer_items = 0
    total_empty_field_count = Counter()

    for path in paths:
        stats = analyze_file(path)
        print_file_stats(path, stats)

        total_samples += stats["sample_count"]
        total_answer_items += stats["answer_item_count"]
        total_empty_field_count.update(stats["empty_field_count"])

    print("=" * 80)
    print("Overall Summary")
    print(f"Total files: {len(paths)}")
    print(f"Total samples: {total_samples}")
    print(f"Total answer items: {total_answer_items}")

    print("\nTotal Empty/N/A fields:")
    for field in FIELDS:
        print(f"  {field}: {total_empty_field_count[field]}")


if __name__ == "__main__":
    main()