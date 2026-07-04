import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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


def read_text(path):
    return Path(path).read_text(encoding="utf-8").strip()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                data.append((line_no, json.loads(line)))
    return data


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


def extract_json_array(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return []


def normalize_items(items):
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        new_item = {}
        for field in FIELDS:
            value = item.get(field, "N/A")
            if value is None or str(value).strip() == "":
                value = "N/A"
            new_item[field] = str(value).strip()
        normalized.append(new_item)
    return normalized


def generate_text(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def build_chunk_messages(system_prompt, user_template, chunk_text):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_template.replace("{chunk_text}", chunk_text),
        },
    ]


def build_merge_messages(system_prompt, user_template, chunk_outputs, max_chunks_for_merge):
    selected = chunk_outputs[:max_chunks_for_merge]

    chunk_text = "\n\n".join(
        [
            f"[Chunk {x['chunk_id']}]\n{x['output']}"
            for x in selected
        ]
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_template.replace("{chunk_outputs}", chunk_text),
        },
    ]


def is_all_na_item(item):
    for field in FIELDS:
        value = str(item.get(field, "N/A")).strip()
        if value not in ["N/A", "NA", "None", "null", ""]:
            return False
    return True


def is_all_na_items(items):
    if not items:
        return True
    return all(is_all_na_item(item) for item in items)


def candidate_score(field, value, chunk_id):
    value = str(value).strip()
    if value in ["", "N/A", "NA", "None", "null"]:
        return -10**9

    score = 0

    # 越靠前的 chunk 越可能包含标题、摘要、研究对象等核心信息
    score += max(0, 100 - chunk_id)

    # 越具体的信息通常越有用
    score += min(len(value), 200) * 0.5

    # 字段特定规则
    low = value.lower()

    if field == "Literature Title":
        # 标题通常较长，且不应像参考文献条目
        if "et al" in low and re.search(r"\b20\d{2}\b", value):
            score -= 50
        score += min(len(value), 180)

    elif field == "Study Area & Country":
        # 地区越具体越好
        score += value.count(",") * 20

    elif field == "Data Year":
        # 出现年份加分
        years = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", value)
        score += len(set(years)) * 30

    elif field == "Accessibility Method":
        method_keywords = [
            "2sfca", "e2sfca", "enhanced", "floating catchment",
            "gravity", "cumulative", "network", "kernel", "hansen"
        ]
        score += sum(30 for kw in method_keywords if kw in low)

    elif field == "Urbanization Rate":
        if "%" in value or "urban" in low or "rural" in low:
            score += 40

    return score


def fallback_rule_merge(chunk_outputs):
    """
    字段级 consensus 聚合：
    1. 解析所有 chunk 的 JSON 输出；
    2. 对每个字段收集非 N/A 候选；
    3. 按 chunk 位置、长度、字段特定规则打分；
    4. 每个字段选择最高分候选。
    """
    candidates = {field: [] for field in FIELDS}

    for c in chunk_outputs:
        chunk_id = c.get("chunk_id", 9999)
        arr = extract_json_array(c.get("output", ""))
        arr = normalize_items(arr)

        for item in arr:
            for field in FIELDS:
                value = item.get(field, "N/A")
                value = str(value).strip()
                if value not in ["", "N/A", "NA", "None", "null"]:
                    candidates[field].append((value, chunk_id))

    merged = {}

    for field in FIELDS:
        vals = candidates[field]
        if not vals:
            merged[field] = "N/A"
            continue

        # 去重，保留最早 chunk
        best_by_value = {}
        for value, chunk_id in vals:
            key = re.sub(r"\W+", "", value.lower())
            if key not in best_by_value or chunk_id < best_by_value[key][1]:
                best_by_value[key] = (value, chunk_id)

        scored = []
        for value, chunk_id in best_by_value.values():
            scored.append((candidate_score(field, value, chunk_id), value, chunk_id))

        scored.sort(key=lambda x: (-x[0], x[2]))
        merged[field] = scored[0][1]

    return [merged]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)

    parser.add_argument("--chunk_system_prompt", default="prompts/chunk_extract_system.txt")
    parser.add_argument("--chunk_user_template", default="prompts/chunk_extract_user_template.txt")
    parser.add_argument("--merge_system_prompt", default="prompts/merge_system.txt")
    parser.add_argument("--merge_user_template", default="prompts/merge_user_template.txt")

    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--max_new_tokens_extract", type=int, default=256)
    parser.add_argument("--max_new_tokens_merge", type=int, default=768)
    parser.add_argument("--max_chunks_for_merge", type=int, default=15)
    parser.add_argument(
        "--aggregation_mode",
        choices=["llm", "consensus", "hybrid"],
        default="hybrid",
        help="Aggregation strategy: llm only, consensus only, or hybrid.",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    chunk_system = read_text(args.chunk_system_prompt)
    chunk_user_template = read_text(args.chunk_user_template)
    merge_system = read_text(args.merge_system_prompt)
    merge_user_template = read_text(args.merge_user_template)

    if "{chunk_text}" not in chunk_user_template:
        raise ValueError("chunk user template must contain {chunk_text}")
    if "{chunk_outputs}" not in merge_user_template:
        raise ValueError("merge user template must contain {chunk_outputs}")

    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("=" * 80)
    print("Loading base model in 4bit, no LoRA")
    print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()

    records = load_jsonl(args.input_file)
    if args.max_samples is not None:
        records = records[:args.max_samples]

    print("=" * 80)
    print("Start baseline chunk inference")
    print("=" * 80)
    print("input:", args.input_file)
    print("output:", args.output_file)
    print("samples:", len(records))

    with open(args.output_file, "w", encoding="utf-8") as out:
        for idx, (line_no, obj) in enumerate(records, 1):
            article_text = obj.get("article_text", "")
            if not article_text:
                article_text = obj["messages"][1]["content"]

            chunks = split_text_by_tokens(
                article_text,
                tokenizer=tokenizer,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            print(f"[{idx}/{len(records)}] line={line_no}, chunks={len(chunks)}")

            chunk_outputs = []

            for chunk_id, chunk_text in enumerate(chunks, 1):
                messages = build_chunk_messages(
                    system_prompt=chunk_system,
                    user_template=chunk_user_template,
                    chunk_text=chunk_text,
                )

                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens_extract,
                )

                chunk_outputs.append({
                    "chunk_id": chunk_id,
                    "output": output,
                })

                print(f"  chunk {chunk_id}/{len(chunks)} done")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            aggregation_used = args.aggregation_mode
            merged_text = ""

            if args.aggregation_mode == "consensus":
                print("  aggregation mode: consensus only")
                merged_items = fallback_rule_merge(chunk_outputs)

            else:
                print(f"  aggregation mode: {args.aggregation_mode}")
                merge_messages = build_merge_messages(
                    system_prompt=merge_system,
                    user_template=merge_user_template,
                    chunk_outputs=chunk_outputs,
                    max_chunks_for_merge=args.max_chunks_for_merge,
                )

                merged_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    messages=merge_messages,
                    max_new_tokens=args.max_new_tokens_merge,
                )

                merged_items = extract_json_array(merged_text)
                merged_items = normalize_items(merged_items)

                if args.aggregation_mode == "llm":
                    if not merged_items:
                        print("  llm merge parse failed, outputing all N/A")
                        merged_items = [{field: "N/A" for field in FIELDS}]
                    elif is_all_na_items(merged_items):
                        print("  llm merge result is all N/A, keep LLM output")

                elif args.aggregation_mode == "hybrid":
                    if not merged_items:
                        print("  merge parse failed, using consensus fallback merge")
                        merged_items = fallback_rule_merge(chunk_outputs)
                        aggregation_used = "hybrid_consensus_fallback"
                    elif is_all_na_items(merged_items):
                        print("  merge result is all N/A, using consensus fallback merge")
                        merged_items = fallback_rule_merge(chunk_outputs)
                        aggregation_used = "hybrid_consensus_fallback"
                    else:
                        aggregation_used = "hybrid_llm_merge"

            assistant_content = json.dumps(
                merged_items,
                ensure_ascii=False,
                indent=2,
            )

            pred_obj = {
                "messages": [
                    obj["messages"][0],
                    obj["messages"][1],
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ],
                "meta": {
                    "source_file": Path(args.input_file).name,
                    "source_line": line_no,
                    "num_chunks": len(chunks),
                    "max_chunks_for_merge": args.max_chunks_for_merge,
                    "model_type": "baseline_base_model_no_lora",
                    "aggregation_mode": args.aggregation_mode,
                    "aggregation_used": aggregation_used,
                },
            }

            out.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
            out.flush()

            print(f"[{idx}/{len(records)}] done, pred_items={len(merged_items)}")

    print("=" * 80)
    print("DONE")
    print("Saved to:", args.output_file)
    print("=" * 80)


if __name__ == "__main__":
    main()
