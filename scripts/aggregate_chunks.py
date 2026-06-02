import argparse
import json
import re
from collections import Counter

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

NA_VALUES = {"n/a", "na", "none", "null", ""}

METHOD_KEYWORDS = [
    "E3SFCA",
    "E2SFCA",
    "2SFCA",
    "3SFCA",
    "Gravity",
    "Hansen",
    "Kernel",
    "Gaussian",
    "Floating catchment",
    "Two-step",
    "Cumulative opportunity",
]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                data.append((line_no, json.loads(line)))
    return data


def normalize_value(value):
    if value is None:
        return ""
    return str(value).strip()


def is_na(value):
    value = normalize_value(value)
    return value.lower() in NA_VALUES


def extract_json_array(text):
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "")
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def looks_like_reference(title):
    title_lower = title.lower()
    if "et al" in title_lower:
        return True
    if re.search(r"\b(18|19|20|21)\d{2}\b", title) and ("," in title or "(" in title):
        return True
    return False


def select_title(candidates):
    if not candidates:
        return "N/A"

    # 论文标题通常出现在前几个chunk，优先选择最早出现的标题
    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[0][0]


def specificity_score(value):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return len(parts) * 10 + len(value)


def select_area(candidates):
    if not candidates:
        return "N/A"
    best = sorted(candidates, key=lambda x: (-specificity_score(x[0]), x[1]))
    return best[0][0]


def select_year(candidates):
    if not candidates:
        return "N/A"
    year_hits = []
    for value, chunk_id in candidates:
        years = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", value)
        for year in years:
            year_hits.append((year, chunk_id))
    if not year_hits:
        return "N/A"
    counts = Counter([y for y, _ in year_hits])
    top_count = max(counts.values())
    top_years = {y for y, c in counts.items() if c == top_count}
    top_candidates = [(y, cid) for y, cid in year_hits if y in top_years]
    top_candidates.sort(key=lambda x: x[1])
    return top_candidates[0][0]


def method_score(value):
    score = 0
    lowered = value.lower()
    for kw in METHOD_KEYWORDS:
        if kw.lower() in lowered:
            score += 1
    return score


def select_method(candidates):
    if not candidates:
        return "N/A"
    best = sorted(
        candidates,
        key=lambda x: (-method_score(x[0]), -len(x[0]), x[1]),
    )
    return best[0][0]


def select_by_length(candidates):
    if not candidates:
        return "N/A"
    best = sorted(candidates, key=lambda x: (-len(x[0]), x[1]))
    return best[0][0]


def select_urbanization(candidates):
    valid = []
    for value, chunk_id in candidates:
        if re.search(r"\d+(\.\d+)?\s*%", value) or "urban" in value.lower() or "???" in value:
            valid.append((value, chunk_id))
    if not valid:
        return "N/A"
    return select_by_length(valid)


def aggregate_record(record):
    chunk_outputs = record.get("chunk_outputs", [])
    candidates = {field: [] for field in FIELDS}

    for chunk in chunk_outputs:
        output_text = chunk.get("output", "")
        parsed = extract_json_array(output_text)
        if not parsed or not isinstance(parsed, list):
            continue
        if not parsed:
            continue
        first = parsed[0]
        if not isinstance(first, dict):
            continue
        values = {field: normalize_value(first.get(field, "")) for field in FIELDS}
        if all(is_na(v) for v in values.values()):
            continue
        title = values.get("Literature Title", "")
        if title and not is_na(title) and looks_like_reference(title):
            continue
        chunk_id = chunk.get("chunk_id", 0)
        for field, value in values.items():
            if not is_na(value):
                candidates[field].append((value, chunk_id))

    aggregated = {
        "Literature Title": select_title(candidates["Literature Title"]),
        "Study Area & Country": select_area(candidates["Study Area & Country"]),
        "Data Year": select_year(candidates["Data Year"]),
        "Accessibility Method": select_method(candidates["Accessibility Method"]),
        "Facility Type": select_by_length(candidates["Facility Type"]),
        "Demand Population": select_by_length(candidates["Demand Population"]),
        "Dist/Time Calc Method": select_by_length(candidates["Dist/Time Calc Method"]),
        "Transport Mode": select_by_length(candidates["Transport Mode"]),
        "Travel Time Period": select_by_length(candidates["Travel Time Period"]),
        "Urbanization Rate": select_urbanization(candidates["Urbanization Rate"]),
    }
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.input_file)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    with open(args.output_file, "w", encoding="utf-8") as out:
        for _, record in records:
            aggregated = aggregate_record(record)
            out.write(json.dumps(aggregated, ensure_ascii=False) + "\n")

    print("DONE:", args.output_file)


if __name__ == "__main__":
    main()
