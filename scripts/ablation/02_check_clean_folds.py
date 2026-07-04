import argparse
import glob
import json
from pathlib import Path
from collections import Counter

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

REQUIRED_ROLES = ["system", "user", "assistant"]
OLD_MARKER = "请严格按照以上要求抽取，只输出JSON数组，不要添加任何额外内容。"


def check_file(path):
    total = 0
    errors = []
    article_lens = []
    answer_items_counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"line {line_no}: invalid jsonl: {e}")
                continue

            if "article_text" not in obj:
                errors.append(f"line {line_no}: missing article_text")

            if "answer" not in obj:
                errors.append(f"line {line_no}: missing answer")

            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                errors.append(f"line {line_no}: invalid messages")
                continue

            roles = [messages[i].get("role") for i in range(3)]
            if roles != REQUIRED_ROLES:
                errors.append(f"line {line_no}: bad roles {roles}")

            user_content = messages[1].get("content", "")
            article_text = obj.get("article_text", "")

            if OLD_MARKER in user_content or OLD_MARKER in article_text:
                errors.append(f"line {line_no}: old marker remains")

            if "{article_text}" in user_content:
                errors.append(f"line {line_no}: template placeholder not replaced")

            if not article_text or len(article_text) < 1000:
                errors.append(f"line {line_no}: article_text too short: {len(article_text)}")

            article_lens.append(len(article_text))

            assistant_content = messages[2].get("content", "")
            try:
                assistant_answer = json.loads(assistant_content)
            except Exception as e:
                errors.append(f"line {line_no}: assistant content invalid JSON: {e}")
                continue

            if not isinstance(assistant_answer, list):
                errors.append(f"line {line_no}: assistant is not JSON array")
                continue

            answer_items_counter[len(assistant_answer)] += 1

            for item_idx, item in enumerate(assistant_answer):
                if not isinstance(item, dict):
                    errors.append(f"line {line_no}: answer item {item_idx} is not dict")
                    continue

                missing = [field for field in FIELDS if field not in item]
                if missing:
                    errors.append(f"line {line_no}: answer item {item_idx} missing fields {missing}")

    print("=" * 80)
    print(path)
    print("samples:", total)
    print("errors:", len(errors))
    if article_lens:
        print(
            "article chars min/avg/max:",
            min(article_lens),
            round(sum(article_lens) / len(article_lens), 1),
            max(article_lens),
        )
    print("answer items per sample:", dict(sorted(answer_items_counter.items())))

    if errors:
        print("First 20 errors:")
        for e in errors[:20]:
            print("  -", e)

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/clean")
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.data_dir) / "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No jsonl files found in {args.data_dir}")

    all_ok = True
    for path in paths:
        all_ok = check_file(path) and all_ok

    print("=" * 80)
    if all_ok:
        print("All clean fold files passed.")
    else:
        print("Some clean fold files have errors.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
