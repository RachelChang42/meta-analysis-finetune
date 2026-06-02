import argparse
import glob
import hashlib
import json
from pathlib import Path


def get_sample_id(obj: dict) -> str:
    """
    用 user 输入内容作为样本唯一标识。
    因为每条样本的 messages[1]['content'] 是文献输入和抽取要求，
    对判断 train/val 是否重叠比较可靠。
    """
    user_content = obj["messages"][1]["content"]
    return hashlib.md5(user_content.encode("utf-8")).hexdigest()


def load_ids(path: Path):
    ids = []
    line_count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_count += 1
            obj = json.loads(line)
            ids.append(get_sample_id(obj))

    return ids, line_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/folds",
        help="Directory containing train_fold_i.jsonl and val_fold_i.jsonl",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        raise SystemExit(1)

    all_val_ids = []
    all_ok = True

    print("=" * 80)
    print("Checking fold files")

    for i in range(1, args.num_folds + 1):
        train_path = data_dir / f"train_fold_{i}.jsonl"
        val_path = data_dir / f"val_fold_{i}.jsonl"

        print("=" * 80)
        print(f"Fold {i}")

        if not train_path.exists():
            print(f"Missing train file: {train_path}")
            all_ok = False
            continue

        if not val_path.exists():
            print(f"Missing val file: {val_path}")
            all_ok = False
            continue

        train_ids, train_lines = load_ids(train_path)
        val_ids, val_lines = load_ids(val_path)

        train_set = set(train_ids)
        val_set = set(val_ids)

        overlap = train_set & val_set

        print(f"Train file: {train_path}")
        print(f"Val file:   {val_path}")
        print(f"Train samples: {train_lines}")
        print(f"Val samples:   {val_lines}")
        print(f"Unique train samples: {len(train_set)}")
        print(f"Unique val samples:   {len(val_set)}")
        print(f"Train/val overlap:    {len(overlap)}")

        if len(train_ids) != len(train_set):
            print("Warning: duplicated samples found inside train file")
            all_ok = False

        if len(val_ids) != len(val_set):
            print("Warning: duplicated samples found inside val file")
            all_ok = False

        if overlap:
            print("Error: train and val overlap exists in this fold")
            all_ok = False

        all_val_ids.extend(val_ids)

    print("=" * 80)
    print("Cross-validation summary")

    all_val_set = set(all_val_ids)

    print(f"Total val samples across folds: {len(all_val_ids)}")
    print(f"Unique val samples across folds: {len(all_val_set)}")

    if len(all_val_ids) != len(all_val_set):
        print("Warning: duplicated samples found across validation folds")
        all_ok = False
    else:
        print("Validation folds are mutually exclusive.")

    discovered_files = sorted(glob.glob(str(data_dir / "*.jsonl")))
    print(f"Total jsonl files in data_dir: {len(discovered_files)}")

    expected_file_count = args.num_folds * 2
    if len(discovered_files) != expected_file_count:
        print(
            f"Warning: expected {expected_file_count} jsonl files, "
            f"but found {len(discovered_files)}"
        )
        all_ok = False

    print("=" * 80)
    print("Summary")

    if all_ok:
        print("All fold checks passed.")
    else:
        print("Some fold checks failed or warnings were found.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()