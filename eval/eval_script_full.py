import ast
import json
import re
import sys
import os
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

FIELDS = [
    "Literature Title", "Study Area & Country", "Data Year", "Accessibility Method",
    "Facility Type", "Demand Population", "Dist/Time Calc Method", "Transport Mode",
    "Travel Time Period", "Urbanization Rate"
]

def parse_json_line_with_flag(raw):
    text = raw.strip()
    if not text:
        return None, False
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text), True
    except Exception:
        start_candidates = [text.find("{"), text.find("[")]
        start_candidates = [i for i in start_candidates if i != -1]
        start = min(start_candidates) if start_candidates else -1
        end = max(text.rfind("}"), text.rfind("]"))
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            try:
                return json.loads(snippet), True
            except Exception:
                try:
                    return ast.literal_eval(snippet), False
                except Exception:
                    return None, False
        return None, False

def parse_json_line(raw):
    obj, _ = parse_json_line_with_flag(raw)
    return obj

def split_json_blocks(text):
    blocks = []
    idx = 0
    while True:
        start = text.find("```json", idx)
        if start == -1:
            break
        start = text.find("\n", start)
        if start == -1:
            break
        end = text.find("```", start)
        if end == -1:
            break
        blocks.append(text[start + 1:end].strip())
        idx = end + 3
    return blocks

def parse_loose_fields(text):
    if not text:
        return None
    result = {}
    for field in FIELDS:
        pattern = rf"\"{re.escape(field)}\"\s*:\s*"
        match = re.search(pattern, text)
        if not match:
            continue
        start = match.end()
        remaining = text[start:]
        if remaining.startswith("\""):
            end_idx = remaining.find("\"", 1)
            if end_idx != -1:
                value = remaining[1:end_idx]
            else:
                value = remaining[1:]
        else:
            end_match = re.search(r"[,}\n]", remaining)
            value = remaining[: end_match.start() if end_match else None]
        result[field] = value.strip()
    return result if result else None

def parse_json_stream(text, warnings=None, source=""):
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    objs = []
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            objs.append(obj)
            idx = end
        except json.JSONDecodeError:
            next_positions = [text.find("{", idx + 1), text.find("[", idx + 1)]
            next_positions = [p for p in next_positions if p != -1]
            if not next_positions:
                break
            if warnings is not None:
                warnings.append(f"{source} 字符{idx}附近无法解析JSON，已跳过")
            idx = min(next_positions)
    return objs

def load_jsonl(path, warnings=None):
    data = defaultdict(list)
    format_flags = {}
    parsed_any = False
    failed_any = False
    skipped_no_title = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_block = False
    block_lines = []
    block_hits = 0
    block_parse_ok = []
    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.lower().startswith("```json"):
            in_block = True
            block_lines = []
            block_hits += 1
            continue
        if in_block and stripped.startswith("```"):
            in_block = False
            block_text = "\n".join(block_lines)
            obj, is_json = parse_json_line_with_flag(block_text)
            if obj is None:
                obj = parse_loose_fields(block_text)
                is_json = False
            if obj is None:
                failed_any = True
                if warnings is not None:
                    warnings.append(f"{os.path.basename(path)} 第{line_no}行json块无法解析")
                block_parse_ok.append(False)
                continue
            block_parse_ok.append(is_json)
            parsed_any = True
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "Literature Title" in item:
                        data[item["Literature Title"]].append(item)
                        format_flags[id(item)] = is_json
                    elif isinstance(item, dict):
                        skipped_no_title += 1
            elif isinstance(obj, dict) and "Literature Title" in obj:
                data[obj["Literature Title"]].append(obj)
                format_flags[id(obj)] = is_json
            elif isinstance(obj, dict):
                skipped_no_title += 1
            continue
        if in_block:
            block_lines.append(line)
            continue

        if not stripped:
            continue
        obj, is_json = parse_json_line_with_flag(line)
        if obj is None:
            obj = parse_loose_fields(line)
            is_json = False
        if obj is None:
            failed_any = True
            if warnings is not None:
                warnings.append(f"{os.path.basename(path)} 第{line_no}行无法解析JSON")
            continue
        parsed_any = True
        if isinstance(obj, str):
            obj, is_json = parse_json_line_with_flag(obj)
            if obj is None:
                obj = parse_loose_fields(line)
                is_json = False
        if isinstance(obj, dict) and "messages" in obj:
            try:
                arr = json.loads(obj["messages"][2]["content"])
                for item in arr:
                    data[item["Literature Title"]].append(item)
                    format_flags[id(item)] = True
            except Exception as e:
                if warnings is not None:
                    warnings.append(f"{os.path.basename(path)} 第{line_no}行messages解析失败: {e}")
        elif isinstance(obj, dict) and "Literature Title" in obj:
            data[obj["Literature Title"]].append(obj)
            format_flags[id(obj)] = is_json
        elif isinstance(obj, dict):
            skipped_no_title += 1
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "Literature Title" in item:
                    data[item["Literature Title"]].append(item)
                elif isinstance(item, dict):
                    skipped_no_title += 1
    if in_block:
        block_text = "\n".join(block_lines)
        obj, is_json = parse_json_line_with_flag(block_text)
        if obj is None:
            obj = parse_loose_fields(block_text)
            is_json = False
        block_parse_ok.append(is_json if obj is not None else False)

    if failed_any and not parsed_any and block_hits == 0:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        block_hits = 0
        for block in split_json_blocks(text):
            block_hits += 1
            obj, is_json = parse_json_line_with_flag(block)
            if obj is None:
                obj = parse_loose_fields(block)
                is_json = False
            if obj is None:
                continue
            block_parse_ok.append(is_json)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "Literature Title" in item:
                        data[item["Literature Title"]].append(item)
                        format_flags[id(item)] = is_json
            elif isinstance(obj, dict) and "Literature Title" in obj:
                data[obj["Literature Title"]].append(obj)
                format_flags[id(obj)] = is_json
        if block_hits == 0:
            objs = parse_json_stream(text, warnings, os.path.basename(path))
        else:
            objs = []
        for obj in objs:
            if isinstance(obj, str):
                obj, is_json = parse_json_line_with_flag(obj)
                if obj is None:
                    obj = parse_loose_fields(obj)
                    is_json = False
            if isinstance(obj, dict) and "messages" in obj:
                try:
                    arr = json.loads(obj["messages"][2]["content"])
                    for item in arr:
                        data[item["Literature Title"]].append(item)
                        format_flags[id(item)] = True
                except Exception as e:
                    if warnings is not None:
                        warnings.append(f"{os.path.basename(path)} messages解析失败: {e}")
            elif isinstance(obj, dict) and "Literature Title" in obj:
                data[obj["Literature Title"]].append(obj)
                format_flags[id(obj)] = is_json
            elif isinstance(obj, dict):
                skipped_no_title += 1
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "Literature Title" in item:
                        data[item["Literature Title"]].append(item)
                    elif isinstance(item, dict):
                        skipped_no_title += 1
    if skipped_no_title and warnings is not None:
        warnings.append(f"{os.path.basename(path)} 有 {skipped_no_title} 条记录缺少 Literature Title")
    if block_parse_ok and warnings is not None:
        warnings.append(f"{os.path.basename(path)} JSON块合规数 {sum(1 for ok in block_parse_ok if ok)}/{len(block_parse_ok)}")
    return data, format_flags, block_parse_ok, block_hits

def check_format(obj):
    if not isinstance(obj, dict):
        return False
    return set(obj.keys()) == set(FIELDS)

def bleu_score(ref, hyp):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)

def to_text(value, normalize=True):
    if not normalize:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def exact_match(a, b, normalize=True):
    return to_text(a, normalize).strip() == to_text(b, normalize).strip()

def loose_match(a, b, normalize=True):
    import re
    def norm(s):
        return re.sub(r'\W+', '', str(to_text(s, normalize)).lower())
    return norm(a) == norm(b)

def f1_score(tp, fp, fn):
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom else 0.0

def token_f1(a, b):
    tokens_a = set(re.findall(r"\w+", to_text(a, True).lower()))
    tokens_b = set(re.findall(r"\w+", to_text(b, True).lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    common = tokens_a & tokens_b
    precision = len(common) / len(tokens_a) if tokens_a else 0.0
    recall = len(common) / len(tokens_b) if tokens_b else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

def field_match(field, gold_value, model_value, normalize=True):
    if field == "Literature Title":
        return exact_match(gold_value, model_value, normalize)
    return token_f1(gold_value, model_value) >= 0.5

def hallucination(ref, hyp, normalize=True):
    ref_text = to_text(ref, normalize).strip()
    hyp_text = to_text(hyp, normalize).strip()
    return (ref_text == "N/A") and (hyp_text not in ["N/A", ""])

def main(model_path, gold_path, warn_path="warnings.txt", normalize=True):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    warnings = []
    model_data, model_format_flags, model_block_ok, model_block_hits = load_jsonl(model_path, warnings)
    gold_data, _, _, _ = load_jsonl(gold_path, warnings)
    model_records = sum(len(v) for v in model_data.values())
    gold_records = sum(len(v) for v in gold_data.values())
    print(f"模型解析到 {len(model_data)} 篇文献, {model_records} 条记录")
    print(f"标注解析到 {len(gold_data)} 篇文献, {gold_records} 条记录")
    
    # 新增：文献输出比例
    ratio = len(model_data) / len(gold_data) if gold_data else 0.0
    print(f"文献输出比例: {ratio:.4f}")
    
    all_titles = set(model_data.keys()) | set(gold_data.keys())
    format_ok, format_total = 0, 0
    halluc_cnt, halluc_total = 0, 0
    bleu_scores = []
    field_tp = {f: 0 for f in FIELDS}
    field_fp = {f: 0 for f in FIELDS}
    field_fn = {f: 0 for f in FIELDS}
    title_correct = 0
    title_total = 0
    area_f1_scores = []
    year_f1_scores = []
    loose_acc = []

    for title in all_titles:
        model_items = model_data.get(title, [])
        gold_items = gold_data.get(title, [])
        if not model_items or not gold_items:
            warnings.append(f"未对齐文献: {title}")
            missing = gold_items if gold_items else model_items
            for _ in missing:
                title_total += 1
                for f in FIELDS:
                    field_fn[f] += 1
                area_f1_scores.append(0.0)
                year_f1_scores.append(0.0)
                loose_acc.append(0.0)
            continue
        for item in model_items:
            format_total += 1
            if model_format_flags.get(id(item), False) and check_format(item):
                format_ok += 1
        if len(gold_items) == 1 and len(model_items) > 1:
            warnings.append(f"模型输出多于标注: {title}")
            gold_items = gold_items * len(model_items)
        if len(gold_items) > 1:
            matched = []
            for m in model_items:
                best, best_score = None, -1
                for g in gold_items:
                    score = sum([loose_match(m.get(f, ""), g.get(f, ""), normalize) for f in FIELDS])
                    if score > best_score:
                        best, best_score = g, score
                matched.append((m, best))
        else:
            matched = list(zip(model_items, gold_items))
        for m, g in matched:
            norm_g = {f: to_text(g.get(f, ""), normalize) for f in FIELDS}
            norm_m = {f: to_text(m.get(f, ""), normalize) for f in FIELDS}
            bleu_scores.append(bleu_score(json.dumps(norm_g, ensure_ascii=False), json.dumps(norm_m, ensure_ascii=False)))
            for f in FIELDS:
                halluc_total += 1
                if hallucination(g.get(f, ""), m.get(f, ""), normalize):
                    halluc_cnt += 1
            for f in FIELDS:
                if field_match(f, g.get(f, ""), m.get(f, ""), normalize):
                    field_tp[f] += 1
                else:
                    field_fp[f] += 1
                    field_fn[f] += 1
            title_total += 1
            if exact_match(g.get("Literature Title", ""), m.get("Literature Title", ""), normalize):
                title_correct += 1
            area_f1_scores.append(token_f1(g.get("Study Area & Country", ""), m.get("Study Area & Country", "")))
            year_f1_scores.append(token_f1(g.get("Data Year", ""), m.get("Data Year", "")))
            loose_acc.append(np.mean([field_match(f, g.get(f, ""), m.get(f, ""), normalize) for f in FIELDS]))
    
    macro_f1 = sum([f1_score(field_tp[f], field_fp[f], field_fn[f]) for f in FIELDS]) / len(FIELDS)
    format_ratio = format_ok / format_total if format_total else 0.0
    print(f"格式合规率: {format_ratio:.4f}")
    print(f"BLEU: {(sum(bleu_scores)/len(bleu_scores)) if bleu_scores else 0:.4f}")
    print(f"宏平均F1: {macro_f1:.4f}")
    print(f"幻觉率: {(halluc_cnt/halluc_total) if halluc_total else 0:.4f}")
    print(f"题目精确匹配: {(title_correct/title_total) if title_total else 0:.4f}")
    print(f"年份F1: {(sum(year_f1_scores)/len(year_f1_scores)) if year_f1_scores else 0:.4f}")
    print(f"地区F1: {(sum(area_f1_scores)/len(area_f1_scores)) if area_f1_scores else 0:.4f}")
    print(f"宽松匹配准确率: {(sum(loose_acc)/len(loose_acc)) if loose_acc else 0:.4f}")
    
    # 新增：逐个字段匹配准确率
    print("\n逐个字段匹配准确率:")
    for f in FIELDS:
        tp = field_tp[f]
        fp = field_fp[f]
        fn = field_fn[f]
        total = tp + fp + fn
        acc = tp / total if total > 0 else 0.0
        print(f"  {f}: {acc:.4f}")
    
    with open(warn_path, "w", encoding="utf-8") as f:
        for w in warnings:
            f.write(w + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python eval_script_finaluse_v2.py <模型输出.jsonl> <标注数据.jsonl> [--no-normalize]")
    else:
        normalize = "--no-normalize" not in sys.argv[3:]
        main(sys.argv[1], sys.argv[2], normalize=normalize)