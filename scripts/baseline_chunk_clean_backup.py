import json
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

            chunks = split_text(user_text, tokenizer, args.chunk_size, args.overlap)
            print(f"[{i}/{len(samples)}] line={line_no}, chunks={len(chunks)}")

            chunk_outputs = []

            for j, chunk in enumerate(chunks, 1):
                messages = [
                    system_msg,
                    {
                        "role": "user",
                        "content": (
                            "下面是一篇医疗可达性文献的一部分。"
                            "只抽取这一部分中明确出现的信息。"
                            "如果没有明确出现，填 N/A。"
                            "严格输出 JSON 数组，不要解释。\n\n"
                            + chunk
                        )
                    }
                ]

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
