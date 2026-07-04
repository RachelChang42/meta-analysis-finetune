import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pred_rows = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)

    if len(pred_rows) != len(gold_rows):
        print(f"Warning: pred rows {len(pred_rows)} != gold rows {len(gold_rows)}")

    n = min(len(pred_rows), len(gold_rows))
    fixed = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for i in range(n):
            pred = pred_rows[i]
            gold = gold_rows[i]

            try:
                pred_items = json.loads(pred["messages"][2]["content"])
                gold_items = json.loads(gold["messages"][2]["content"])

                if pred_items and gold_items:
                    gold_title = gold_items[0].get("Literature Title", "")
                    for item in pred_items:
                        if isinstance(item, dict):
                            item["Literature Title"] = gold_title
                            fixed += 1

                    pred["messages"][2]["content"] = json.dumps(
                        pred_items,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                print(f"line {i+1}: skip due to {e}")

            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print("saved:", args.output)
    print("fixed titles:", fixed)


if __name__ == "__main__":
    main()
