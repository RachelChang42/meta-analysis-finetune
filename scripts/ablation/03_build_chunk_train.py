import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def split_text_by_tokens(text, tokenizer, chunk_size, overlap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(1, chunk_size - overlap)

    chunks = []
    for start in range(0, len(ids), step):
        part = ids[start:start + chunk_size]
        chunk_text = tokenizer.decode(part, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if start + chunk_size >= len(ids):
            break

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="clean train fold jsonl")
    parser.add_argument("--output", required=True, help="chunk train output jsonl")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--system_prompt", default="prompts/chunk_train_system.txt")
    parser.add_argument("--user_template", default="prompts/chunk_train_user_template.txt")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    system_prompt = read_text(args.system_prompt)
    user_template = read_text(args.user_template)

    if "{chunk_text}" not in user_template:
        raise ValueError("chunk train user template must contain {chunk_text}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0
    min_chunks = None
    max_chunks = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            total_docs += 1

            article_text = obj.get("article_text", "")
            if not article_text:
                raise ValueError(f"line {line_no}: missing article_text")

            # 这里是文献级人工标注，因此 chunk train 是 weakly supervised。
            assistant_content = obj["messages"][2]["content"]

            chunks = split_text_by_tokens(
                article_text,
                tokenizer=tokenizer,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            total_chunks += len(chunks)
            min_chunks = len(chunks) if min_chunks is None else min(min_chunks, len(chunks))
            max_chunks = max(max_chunks, len(chunks))

            for chunk_id, chunk_text in enumerate(chunks, start=1):
                user_content = user_template.replace("{chunk_text}", chunk_text)

                chunk_obj = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content,
                        },
                    ],
                    "meta": {
                        "source_file": Path(args.input).name,
                        "source_line": line_no,
                        "chunk_id": chunk_id,
                        "num_chunks": len(chunks),
                        "chunk_size": args.chunk_size,
                        "overlap": args.overlap,
                        "supervision_type": "weak_document_label_copied_to_chunks",
                    },
                }

                fout.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")

    avg_chunks = round(total_chunks / total_docs, 2) if total_docs else 0

    print("input:", args.input)
    print("output:", args.output)
    print("documents:", total_docs)
    print("total chunks:", total_chunks)
    print("chunks per doc min/avg/max:", min_chunks, avg_chunks, max_chunks)
    print("supervision:", "weak_document_label_copied_to_chunks")


if __name__ == "__main__":
    main()
