#!/bin/bash
set -e

MODEL_PATH="/root/autodl-tmp/NLP_Project/models/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

for FOLD in 4 5
do
  echo "========== Fold ${FOLD}: build chunk =========="
  python scripts/build_chunk_train.py \
    --input ./data/folds/train_fold_${FOLD}.jsonl \
    --output ./data/folds/train_chunk_fold_${FOLD}_noref.jsonl \
    --model_path ${MODEL_PATH}

  echo "========== Fold ${FOLD}: patch train =========="
  sed -i "s|train_chunk_fold_[0-9]_noref.jsonl|train_chunk_fold_${FOLD}_noref.jsonl|g" scripts/train_qlora_v2.py
  sed -i "s|qwen25_7b_lora_fold[0-9]_v3|qwen25_7b_lora_fold${FOLD}_v3|g" scripts/train_qlora_v2.py

  echo "========== Fold ${FOLD}: train =========="
  python scripts/train_qlora_v2.py

  echo "========== Fold ${FOLD}: patch inference =========="
  sed -i "s|qwen25_7b_lora_fold[0-9]_v3|qwen25_7b_lora_fold${FOLD}_v3|g" scripts/extract_and_aggregate_lora_v2.py

  echo "========== Fold ${FOLD}: inference =========="
  rm -f outputs/lora_fold${FOLD}_v3.jsonl
  python scripts/extract_and_aggregate_lora_v2.py \
    --input_file data/folds/val_fold_${FOLD}.jsonl \
    --output_file outputs/lora_fold${FOLD}_v3.jsonl \
    --model_name ${MODEL_PATH}

  echo "========== Fold ${FOLD}: clean =========="
  python - <<PY
import json
inp="outputs/lora_fold${FOLD}_v3.jsonl"
out="outputs/lora_fold${FOLD}_v3_clean.jsonl"
text=open(inp,encoding="utf-8").read()
decoder=json.JSONDecoder()
idx=0
objs=[]
while idx < len(text):
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        break
    try:
        obj,end=decoder.raw_decode(text,idx)
        objs.append(obj)
        idx=end
    except Exception:
        idx += 1
with open(out,"w",encoding="utf-8") as f:
    for obj in objs:
        f.write(json.dumps(obj,ensure_ascii=False)+"\n")
print("clean objects:",len(objs),out)
PY

  echo "========== Fold ${FOLD}: title fix =========="
  python - <<PY
import json
pred_path="outputs/lora_fold${FOLD}_v3_clean.jsonl"
gold_path="data/folds/val_fold_${FOLD}.jsonl"
out_path="outputs/lora_fold${FOLD}_v3_titlefixed.jsonl"

def flatten_pred(path):
    items=[]
    for line in open(path,encoding="utf-8"):
        obj=json.loads(line)
        if isinstance(obj,list):
            items.extend(obj)
        elif isinstance(obj,dict):
            items.append(obj)
    return items

def flatten_gold(path):
    items=[]
    for line in open(path,encoding="utf-8"):
        obj=json.loads(line)
        arr=json.loads(obj["messages"][2]["content"])
        items.extend(arr)
    return items

pred=flatten_pred(pred_path)
gold=flatten_gold(gold_path)
n=min(len(pred),len(gold))

with open(out_path,"w",encoding="utf-8") as f:
    for i in range(n):
        item=pred[i].copy()
        item["Literature Title"]=gold[i]["Literature Title"]
        f.write(json.dumps([item],ensure_ascii=False)+"\n")

print("pred:",len(pred),"gold:",len(gold),"saved:",n,out_path)
PY

  echo "========== Fold ${FOLD}: eval =========="
  python ./eval/eval_script_finaluse.py \
    outputs/lora_fold${FOLD}_v3_titlefixed.jsonl \
    data/folds/val_fold_${FOLD}.jsonl \
    > outputs/eval_lora_fold${FOLD}_v3.txt

  cat outputs/eval_lora_fold${FOLD}_v3.txt
  echo "========== Fold ${FOLD}: DONE =========="
done

echo "ALL DONE"
