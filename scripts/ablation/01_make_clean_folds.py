import argparse
import glob
import json
import re
from pathlib import Path


ARTICLE_MARKER = "请严格按照以上要求抽取，只输出JSON数组，不要添加任何额外内容。"

TAIL_PATTERNS = [
    r"\n\s*references\s*\n",
    r"\n\s*bibliography\s*\n",
    r"\n\s*acknowledgements\s*\n",
    r"\n\s*acknowledgments\s*\n",
    r"\n\s*funding\s*\n",
    r"\n\s*author details\s*\n",
    r"\n\s*competing interests\s*\n",
    r"\n\s*conflicts of interest\s*\n",
    r"\n\s*data availability statement\s*\n",
    r"\n\s*availability of data and materials\s*\n",
    r"\n\s*ethics approval",
    r"\n\s*consent for publication",
    r"\n\s*authors'? contributions",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def remove_tail_sections(text: str) -> str:
    cut_positions = []
    for pattern in TAIL_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        return text[: min(cut_positions)].strip()

    return text.strip()


def extract_article_text(user_content: str):
    used_marker = ARTICLE_MARKER in user_content

    if used_marker:
        article_text = user_content.split(ARTICLE_MARKER, 1)[1]
    else:
        article_text = user_content

    article_text = article_text.strip()
    article_text = remove_tail_sections(article_text)

    return article_text, used_marker


def normalize_answer(assistant_content: str):
    answer = json.loads(assistant_content)
    if not isinstance(answer, list):
        raise ValueError("assistant content is not a JSON array")

    answer_text = json.dumps(answer, ensure_ascii=False, indent=2)
    return answer_text, answer


def process_file(input_path: Path, output_path: Path, system_prompt: str, user_template: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    used_marker_count = 0
    no_marker_count = 0

    min_article_chars = None
    max_article_chars = 0
    total_article_chars = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            obj = json.loads(line)
            messages = obj["messages"]

            if len(messages) < 3:
                raise ValueError(f"{input_path} line {line_no}: messages length < 3")

            raw_user = messages[1]["content"]
            article_text, used_marker = extract_article_text(raw_user)

            if used_marker:
                used_marker_count += 1
            else:
                no_marker_count += 1

            article_chars = len(article_text)
            min_article_chars = article_chars if min_article_chars is None else min(min_article_chars, article_chars)
            max_article_chars = max(max_article_chars, article_chars)
            total_article_chars += article_chars

            assistant_content, answer = normalize_answer(messages[2]["content"])
            clean_user = user_template.replace("{article_text}", article_text)

            clean_obj = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": clean_user,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ],
                "article_text": article_text,
                "answer": answer,
                "meta": {
                    "source_file": input_path.name,
                    "source_line": line_no,
                    "article_chars": article_chars,
                    "used_article_marker": used_marker,
                },
            }

            fout.write(json.dumps(clean_obj, ensure_ascii=False) + "\n")

    avg_article_chars = round(total_article_chars / total, 1) if total else 0

    print(f"{input_path} -> {output_path}")
    print(f"  samples: {total}")
    print(f"  used marker: {used_marker_count}")
    print(f"  no marker: {no_marker_count}")
    print(f"  article chars min/avg/max: {min_article_chars}/{avg_article_chars}/{max_article_chars}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/clean")
    parser.add_argument("--system_prompt", default="prompts/extract_system.txt")
    parser.add_argument("--user_template", default="prompts/extract_user_template.txt")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    system_prompt = read_text(Path(args.system_prompt))
    user_template = read_text(Path(args.user_template))

    if "{article_text}" not in user_template:
        raise ValueError("user_template must contain {article_text}")

    paths = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No jsonl files found in {raw_dir}")

    for p in paths:
        input_path = Path(p)
        output_path = out_dir / input_path.name
        process_file(input_path, output_path, system_prompt, user_template)

    print("DONE")


if __name__ == "__main__":
    main()
