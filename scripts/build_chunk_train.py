import json
import argparse
import re
from transformers import AutoTokenizer


def remove_references(text):
    patterns = [
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

    cut_positions = []
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        return text[:min(cut_positions)]

    return text


def split_text(text, tokenizer, chunk_size, overlap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(ids), step):
        part = ids[start:start + chunk_size]
        chunk = tokenizer.decode(part, skip_special_tokens=True)
        chunks.append(chunk)

        if start + chunk_size >= len(ids):
            break

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    total_papers = 0
    total_chunks = 0
    removed_ref_count = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)
            messages = item["messages"]

            system_msg = messages[0]
            user_msg = messages[1]
            assistant_msg = messages[2]

            original_text = user_msg["content"]
            user_text = remove_references(original_text)

            if len(user_text) < len(original_text):
                removed_ref_count += 1

            answer_text = assistant_msg["content"]

            chunks = split_text(
                user_text,
                tokenizer,
                args.chunk_size,
                args.overlap
            )

            total_papers += 1
            total_chunks += len(chunks)

            for i, chunk in enumerate(chunks):
                new_item = {
                    "messages": [
                        system_msg,
                        {
                            "role": "user",
                            "content": chunk
                        },
                        {
                            "role": "assistant",
                            "content": answer_text
                        }
                    ]
                }
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print("原始论文数:", total_papers)
    print("去掉参考文献/尾部信息的论文数:", removed_ref_count)
    print("生成chunk训练样本数:", total_chunks)
    print("平均每篇chunk数:", total_chunks / total_papers)


if __name__ == "__main__":
    main()
