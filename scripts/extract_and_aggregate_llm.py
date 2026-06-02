import argparse
import json
import re
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                data.append((line_no, json.loads(line)))
    return data


def split_text(text, tokenizer, chunk_size, overlap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(ids), step):
        part = ids[start:start + chunk_size]
        chunks.append(tokenizer.decode(part, skip_special_tokens=True))
        if start + chunk_size >= len(ids):
            break
    return chunks


def remove_references(text):
    """
    删除论文末尾参考文献部分
    """
    pattern = re.compile(
        r"(?im)^\s*(references|bibliography|参考文献|参考资料)\s*[:：]?\s*$"
    )

    match = pattern.search(text)

    if match:
        return text[:match.start()]

    return text


def build_extract_messages(chunk_text):
    return [
        {
            "role": "system",
            "content": "你是一名医疗可达性文献信息抽取助手。只根据给定文本抽取信息，缺失填N/A，严格输出JSON数组。",
        },
        {
            "role": "user",
            "content": (
                "下面是一篇医疗可达性文献的一部分。\n\n"
                "请只提取以下10个字段：\n"
                "Literature Title\n"
                "Study Area & Country\n"
                "Data Year\n"
                "Accessibility Method\n"
                "Facility Type\n"
                "Demand Population\n"
                "Dist/Time Calc Method\n"
                "Transport Mode\n"
                "Travel Time Period\n"
                "Urbanization Rate\n\n"
                "未出现的信息填写 N/A。\n\n"
                "严格按照下面格式输出：\n"
                "[\n"
                "  {\n"
                "    \"Literature Title\": \"\",\n"
                "    \"Study Area & Country\": \"\",\n"
                "    \"Data Year\": \"\",\n"
                "    \"Accessibility Method\": \"\",\n"
                "    \"Facility Type\": \"\",\n"
                "    \"Demand Population\": \"\",\n"
                "    \"Dist/Time Calc Method\": \"\",\n"
                "    \"Transport Mode\": \"\",\n"
                "    \"Travel Time Period\": \"\",\n"
                "    \"Urbanization Rate\": \"\"\n"
                "  }\n"
                "]\n\n"
                "不要输出其他字段。\n"
                "不要解释。\n"
                "不要总结。\n"
                "只输出JSON。\n\n"
                + chunk_text
            ),
        },
    ]


def build_merge_messages(chunk_outputs):
    text = "\n\n".join(
        [f"[Chunk {c['chunk_id']}]\n{c['output']}" for c in chunk_outputs]
    )

    return [
        {
            "role": "system",
            "content": (
                "你是医疗文献信息融合专家。"
                "负责将多个chunk抽取结果合并为唯一JSON。"
                "冲突选择最具体最可靠信息，只输出JSON。"
            ),
        },
        {
            "role": "user",
            "content": f"""
请融合以下chunk抽取结果，输出唯一JSON：

字段：
- Literature Title
- Study Area & Country
- Data Year
- Accessibility Method
- Facility Type
- Demand Population
- Dist/Time Calc Method
- Transport Mode
- Travel Time Period
- Urbanization Rate

要求：
1. 去重
2. 冲突取最可信
3. 缺失填 N/A
4. 只输出JSON
5. Urbanization Rate可以是描述性；

内容：
{text}
""",
        },
    ]


def generate_text(tokenizer, model, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    records = load_jsonl(args.input_file)
    if args.start_index:
        records = records[args.start_index:]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    with open(args.output_file, "a", encoding="utf-8") as out:
        for _, record in records:
            messages = record.get("messages", [])
            if len(messages) < 2:
                continue
            user_text = messages[1].get("content", "")
            marker = "请严格按照以上要求抽取，只输出JSON数组，不要添加任何额外内容。"
            if marker in user_text:
                user_text = user_text.split(marker, 1)[1]
            user_text = remove_references(user_text)

            chunks = split_text(user_text, tokenizer, args.chunk_size, args.overlap)
            chunk_outputs = []
            for chunk_id, chunk in enumerate(chunks, 1):
                extract_messages = build_extract_messages(chunk)
                output = generate_text(tokenizer, model, extract_messages, args.max_new_tokens)
                chunk_outputs.append({"chunk_id": chunk_id, "output": output})

            merge_messages = build_merge_messages(chunk_outputs)
            merged = generate_text(tokenizer, model, merge_messages, args.max_new_tokens)

            out.write(merged + "\n")
            out.flush()

    print("DONE:", args.output_file)


if __name__ == "__main__":
    main()
