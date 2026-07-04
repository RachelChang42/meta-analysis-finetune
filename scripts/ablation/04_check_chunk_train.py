import argparse
import glob
import json
from pathlib import Path
from collections import Counter

REQUIRED_ROLES = ["system", "user", "assistant"]

def check_file(path):
    total = 0
    errors = []
    num_chunks_counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"line {line_no}: invalid json: {e}")
                continue

            messages = obj.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 3:
                errors.append(f"line {line_no}: invalid messages")
                continue

            roles = [messages[i].get("role") for i in range(3)]
            if roles != REQUIRED_ROLES:
                errors.append(f"line {line_no}: bad roles {roles}")

            system = messages[0].get("content", "")
            user = messages[1].get("content", "")
            assistant = messages[2].get("content", "")

            if "文献片段" not in system and "文献片段" not in user:
                errors.append(f"line {line_no}: chunk train prompt missing")

            if "训练目标" not in user:
                errors.append(f"line {line_no}: training objective missing")

            if "{chunk_text}" in user:
                errors.append(f"line {line_no}: chunk placeholder not replaced")

            if "文献内容：" in user:
                errors.append(f"line {line_no}: full-document prompt appears in chunk train")

            try:
                ans = json.loads(assistant)
                if not isinstance(ans, list):
                    errors.append(f"line {line_no}: assistant is not JSON array")
            except Exception as e:
                errors.append(f"line {line_no}: assistant invalid json: {e}")

            meta = obj.get("meta", {})
            if meta.get("supervision_type") != "weak_document_label_copied_to_chunks":
                errors.append(f"line {line_no}: missing or wrong supervision_type")

            if "chunk_id" not in meta or "num_chunks" not in meta:
                errors.append(f"line {line_no}: missing chunk metadata")
            else:
                num_chunks_counter[meta["num_chunks"]] += 1

    print("=" * 80)
    print(path)
    print("samples:", total)
    print("errors:", len(errors))
    if num_chunks_counter:
        print("num_chunks distribution top10:", num_chunks_counter.most_common(10))

    if errors:
        print("First 20 errors:")
        for e in errors[:20]:
            print("  -", e)

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/chunk_train")
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.data_dir) / "train_chunk_fold_*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No train_chunk_fold_*.jsonl found in {args.data_dir}")

    all_ok = True
    for path in paths:
        all_ok = check_file(path) and all_ok

    print("=" * 80)
    if all_ok:
        print("All chunk train files passed.")
    else:
        print("Some chunk train files have errors.")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
