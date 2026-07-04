import argparse
import json
from pathlib import Path


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


def normalize_item(item):
    out = {}
    for f in FIELDS:
        v = item.get(f, "N/A") if isinstance(item, dict) else "N/A"
        if v is None or str(v).strip() == "":
            v = "N/A"
        out[f] = str(v).strip()
    return out


def convert_one_file(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_items = 0
    failed = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1

            try:
                obj = json.loads(line)

                # 标准 messages 格式
                if isinstance(obj, dict) and "messages" in obj:
                    content = obj["messages"][2]["content"]
                    items = json.loads(content)

                # 如果本来就是 list
                elif isinstance(obj, list):
                    items = obj

                # 如果本来就是单条 dict
                elif isinstance(obj, dict):
                    items = [obj]

                else:
                    items = []

                if isinstance(items, dict):
                    items = [items]

                for item in items:
                    if isinstance(item, dict):
                        clean_item = normalize_item(item)
                        fout.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                        total_items += 1

            except Exception as e:
                failed += 1
                print(f"[WARN] {input_path} line {line_no} failed: {e}")

    print(f"[OK] {input_path} -> {output_path} | lines={total_lines}, items={total_items}, failed={failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="output/ablation")
    parser.add_argument("--output_root", default="output/ablation_fields")
    parser.add_argument("--skip_debug", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    paths = sorted(input_root.rglob("*.jsonl"))

    if args.skip_debug:
        paths = [p for p in paths if "debug" not in p.name.lower()]

    if not paths:
        print("No jsonl files found.")
        return

    for p in paths:
        rel = p.relative_to(input_root)
        out_name = rel.with_name(rel.stem + "_fields.jsonl")
        out_path = output_root / out_name
        convert_one_file(p, out_path)

    print("=" * 80)
    print("DONE")
    print("output_root:", output_root)


if __name__ == "__main__":
    main()
