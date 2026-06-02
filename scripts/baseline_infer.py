import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append((line_no, obj))
    return samples


def extract_json_array(text):
    """
    尽量从模型输出中提取 JSON 数组。
    如果模型严格输出 JSON 数组，就直接返回。
    如果模型输出了额外解释文字，就尝试截取第一个 [...]。
    如果仍然失败，则返回原始文本，后续测评时会暴露格式问题。
    """
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return text


def build_prompt_messages(obj):
    messages = obj["messages"]

    if len(messages) < 2:
        raise ValueError("messages length < 2")

    if messages[0].get("role") != "system":
        raise ValueError("messages[0] is not system")

    if messages[1].get("role") != "user":
        raise ValueError("messages[1] is not user")

    # baseline 推理只允许使用 system + user
    # 不能把 messages[2] 的人工标注答案喂给模型
    system_msg = messages[0]
    user_msg = {
        "role": "user",
        "content": messages[1]["content"]
        + "\n\n请严格只输出一个 JSON 数组，数组元素必须是对象。"
        + "不要输出解释文字，不要输出 Markdown，不要输出代码块，不要输出感叹号。"
        + "如果某个字段没有信息，请填 \"N/A\"。"
    }

    return [system_msg, user_msg]


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

    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1536)

    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

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
    tokenizer.truncation_side = "right"

    print("=" * 80)
    print("Loading baseline model in 4bit")
    print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    
    config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    cache_dir=args.cache_dir,
    )

    config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    }

    model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=args.cache_dir,
    )

    model.eval()

    samples = load_jsonl(args.input_file)

    print("=" * 80)
    print("Start baseline inference")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Samples: {len(samples)}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, (line_no, obj) in enumerate(samples, start=1):
            prompt_messages = build_prompt_messages(obj)

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=False,
            )
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            assistant_content = extract_json_array(generated_text)

            pred_obj = {
                "messages": [
                    obj["messages"][0],
                    obj["messages"][1],
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ]
            }

            out_f.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"[{idx}/{len(samples)}] line {line_no} done")

    print("=" * 80)
    print("Baseline inference finished")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()