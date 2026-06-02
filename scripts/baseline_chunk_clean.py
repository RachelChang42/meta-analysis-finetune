import json
import re
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1)
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    samples = load_jsonl(args.input_file)[:args.max_samples]
    print("samples:", len(samples))

    with open(args.output_file, "w", encoding="utf-8") as out:
        for i, (line_no, obj) in enumerate(samples, 1):
            system_msg = obj["messages"][0]
            user_text = obj["messages"][1]["content"]

            marker = "请严格按照以上要求抽取，只输出JSON数组，不要添加任何额外内容。"

            if marker in user_text:
                user_text = user_text.split(marker, 1)[1]
            pattern = re.compile(
                r"(?im)^\s*(references|bibliography|参考文献|参考资料)\s*[:：]?\s*$"
            )
            match = pattern.search(user_text)
            if match:
                user_text = user_text[:match.start()]
            chunks = split_text(user_text, tokenizer, args.chunk_size, args.overlap)
            print(f"[{i}/{len(samples)}] line={line_no}, chunks={len(chunks)}")

            chunk_outputs = []

            for j, chunk in enumerate(chunks, 1):
                messages = [
                    {
                        "role": "system",
                        "content": "你是一名医疗可达性文献信息抽取助手。只根据给定文本抽取信息，缺失填N/A，严格输出JSON数组。"
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

                            + chunk
                        )
                    }                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.chunk_size + 512
                ).to(model.device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                gen = outputs[0][inputs["input_ids"].shape[-1]:]
                text = tokenizer.decode(gen, skip_special_tokens=True).strip()

                chunk_outputs.append({
                    "chunk_id": j,
                    "output": text
                })

                print(f"  chunk {j}/{len(chunks)} done")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            result = {
                "line_no": line_no,
                "num_chunks": len(chunks),
                "chunk_outputs": chunk_outputs
            }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    print("DONE:", args.output_file)

if __name__ == "__main__":
    main()
