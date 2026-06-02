import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


EMPTY_VALUES = {"", "N/A", "NA", "None", "null", None}


def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            samples.append((line_no, json.loads(line)))
    return samples


def safe_json_array(text):
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return []


def normalize_item(item):
    if not isinstance(item, dict):
        return None

    normalized = {}

    for field in FIELDS:
        value = item.get(field, "N/A")
        if value is None:
            value = "N/A"
        normalized[field] = value

    return normalized


def is_empty(value):
    return value is None or str(value).strip() in EMPTY_VALUES


def norm_title(title):
    return re.sub(r"\W+", "", str(title).lower())


def merge_items(items):
    """
    简单合并策略：
    1. 按 Literature Title 分组；
    2. 同一标题下，优先保留非空字段；
    3. 如果标题缺失，则放入 unknown 组。
    """
    groups = {}

    for item in items:
        item = normalize_item(item)
        if item is None:
            continue

        title = item.get("Literature Title", "N/A")
        key = norm_title(title)

        if not key:
            key = f"unknown_{len(groups)}"

        if key not in groups:
            groups[key] = {field: "N/A" for field in FIELDS}

        for field in FIELDS:
            old_value = groups[key].get(field, "N/A")
            new_value = item.get(field, "N/A")

            if is_empty(old_value) and not is_empty(new_value):
                groups[key][field] = new_value

    merged = list(groups.values())

    return merged


def split_tokens(token_ids, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunks.append(token_ids[start:end])

        if end >= len(token_ids):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunk_prompt(system_content, instruction_text, chunk_text, chunk_idx, total_chunks):
    user_content = f"""
下面是同一篇医疗可达性研究文献的第 {chunk_idx}/{total_chunks} 个片段。
请只根据当前片段抽取能确定的信息。

如果当前片段没有出现某个字段，请填 "N/A"。
如果当前片段没有足够信息形成完整条目，请输出 []。
请严格只输出 JSON 数组，不要输出解释文字，不要 Markdown，不要代码块。

字段必须严格为：
{json.dumps(FIELDS, ensure_ascii=False, indent=2)}

原始抽取要求摘要：
{instruction_text}

当前文献片段：
{chunk_text}
""".strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def generate_one(model, tokenizer, messages, max_new_tokens):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=False,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/NLP_Project/models/huggingface",
    )

    parser.add_argument("--instruction_tokens", type=int, default=1200)
    parser.add_argument("--chunk_tokens", type=int, default=2200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=768)

    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = args.cache_dir

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("=" * 80)
    print("Loading baseline model in 4bit")
    print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    samples = load_jsonl(args.input_file)

    print("=" * 80)
    print("Start chunk baseline inference")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Samples: {len(samples)}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for sample_idx, (line_no, obj) in enumerate(samples, start=1):
            messages = obj["messages"]

            system_content = messages[0]["content"]
            user_content = messages[1]["content"]

            user_ids = tokenizer(
                user_content,
                add_special_tokens=False,
            )["input_ids"]

            instruction_ids = user_ids[: args.instruction_tokens]
            article_ids = user_ids[args.instruction_tokens :]

            instruction_text = tokenizer.decode(
                instruction_ids,
                skip_special_tokens=True,
            )

            if not article_ids:
                article_ids = user_ids

            chunks = split_tokens(
                article_ids,
                chunk_size=args.chunk_tokens,
                overlap=args.chunk_overlap,
            )

            print(f"[{sample_idx}/{len(samples)}] line {line_no}: chunks={len(chunks)}")

            all_items = []

            for chunk_idx, chunk_ids in enumerate(chunks, start=1):
                chunk_text = tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                )

                chunk_messages = build_chunk_prompt(
                    system_content=system_content,
                    instruction_text=instruction_text,
                    chunk_text=chunk_text,
                    chunk_idx=chunk_idx,
                    total_chunks=len(chunks),
                )

                generated_text = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    messages=chunk_messages,
                    max_new_tokens=args.max_new_tokens,
                )

                items = safe_json_array(generated_text)
                all_items.extend(items)

                print(
                    f"  chunk {chunk_idx}/{len(chunks)}: "
                    f"generated_items={len(items)}"
                )

            merged_items = merge_items(all_items)

            assistant_content = json.dumps(
                merged_items,
                ensure_ascii=False,
                indent=2,
            )

            pred_obj = {
                "messages": [
                    messages[0],
                    messages[1],
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ]
            }

            out_f.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"[{sample_idx}/{len(samples)}] line {line_no} done, "
                f"merged_items={len(merged_items)}"
            )

    print("=" * 80)
    print("Chunk baseline inference finished")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()