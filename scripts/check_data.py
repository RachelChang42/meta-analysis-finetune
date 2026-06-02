import argparse
import glob
import json
from pathlib import Path


REQUIRED_ROLES = ["system", "user", "assistant"]

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

def check_one_file(path: str) -> bool:
    total_lines = 0
    valid_lines = 0
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                errors.append(f"line {line_no}: empty line")
                continue

            total_lines += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSONL line: {e}")
                continue

            if not isinstance(obj, dict):
                errors.append(f"line {line_no}: top-level object is not a dict")
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list):
                errors.append(f"line {line_no}: missing or invalid 'messages'")
                continue

            if len(messages) < 3:
                errors.append(f"line {line_no}: messages length < 3")
                continue

            roles = []
            for i in range(3):
                if not isinstance(messages[i], dict):
                    errors.append(f"line {line_no}: messages[{i}] is not a dict")
                    break
                roles.append(messages[i].get("role"))

            if roles != REQUIRED_ROLES:
                errors.append(
                    f"line {line_no}: role order should be {REQUIRED_ROLES}, got {roles}"
                )
                continue

            for i in range(3):
                content = messages[i].get("content")
                if not isinstance(content, str):
                    errors.append(f"line {line_no}: messages[{i}]['content'] is not a string")

            assistant_content = messages[2].get("content")
            if not isinstance(assistant_content, str):
                continue

            try:
                answer = json.loads(assistant_content)
            except json.JSONDecodeError as e:
                errors.append(
                    f"line {line_no}: assistant content is not valid JSON string: {e}"
                )
                continue

            if not isinstance(answer, list):
                errors.append(f"line {line_no}: assistant content should be a JSON list")
                continue

            if len(answer) == 0:
                errors.append(f"line {line_no}: assistant answer list is empty")
                continue

            for item_idx, item in enumerate(answer):
                if not isinstance(item, dict):
                    errors.append(
                        f"line {line_no}: answer item {item_idx} is not a dict"
                    )
                    continue

                missing_fields = [field for field in FIELDS if field not in item]
                extra_fields = [field for field in item.keys() if field not in FIELDS and field not in OPTIONAL_FIELDS]
                if missing_fields:
                    errors.append(
                        f"line {line_no}: answer item {item_idx} missing fields: {missing_fields}"
                    )

                if extra_fields:
                    errors.append(
                        f"line {line_no}: answer item {item_idx} extra fields: {extra_fields}"
                    )

            valid_lines += 1

    print("=" * 80)
    print(f"File: {path}")
    print(f"Total non-empty lines: {total_lines}")
    print(f"Valid lines: {valid_lines}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nFirst 20 errors:")
        for err in errors[:20]:
            print(f"  - {err}")

        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")

    return len(errors) == 0


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

    all_ok = True

    for path in paths:
        ok = check_one_file(path)
        all_ok = all_ok and ok

    print("=" * 80)
    print("Summary")

    if all_ok:
        print("All files passed data format check.")
    else:
        print("Some files failed data format check.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()