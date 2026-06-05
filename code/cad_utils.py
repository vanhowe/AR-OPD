import json
import re
from collections import Counter


_ACTION_RE = re.compile(r"Action:\s*([^\n\r]+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*({.*?})", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
_OPTION_NUMERIC_RE = re.compile(r"^\s*([A-D])\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\(?\d+\)?[.)])\s*")
_COUNTING_BUCKETS = {
    "molar_weight",
    "heavy_atoms",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
}
_KEYWORD_LINES_BY_BUCKET = {
    "molar_weight": ("total", "sum", "mass", "g/mol", "molar"),
    "heavy_atoms": ("heavy atom", "heavy atoms", "count", "total"),
    "h_bond_donors": ("donor", "donors", "count", "total"),
    "h_bond_acceptors": ("acceptor", "acceptors", "count", "total"),
    "rotatable_bonds": ("rotatable", "count", "total"),
}
_SUMMARY_LINE_HINTS = ("therefore", "thus", "overall", "final", "answer", "hence", "so ")


def _normalize_scalar(value):
    if isinstance(value, str):
        stripped = value.strip()
        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped.lower()
    return value


def _normalize_obj(value):
    if isinstance(value, dict):
        return {key: _normalize_obj(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_obj(item) for item in value]
    return _normalize_scalar(value)


def extract_science_answer(text):
    if not text:
        return None
    match = _ANSWER_RE.search(text)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer or None


def _extract_reasoning(text):
    if not text:
        return ""
    match = _REASONING_RE.search(text)
    if match:
        return match.group(1).strip()
    if "<answer>" in text:
        return text.split("<answer>", 1)[0].strip()
    return text.strip()


def _infer_science_bucket(question_text):
    lower = (question_text or "").lower()
    if "molar weight" in lower or "molecular weight" in lower or "g/mol" in lower:
        return "molar_weight"
    if "heavy atoms" in lower:
        return "heavy_atoms"
    if "hydrogen bond donors" in lower:
        return "h_bond_donors"
    if "hydrogen bond acceptors" in lower:
        return "h_bond_acceptors"
    if "rotatable bonds" in lower:
        return "rotatable_bonds"
    if "logd" in lower or "log p" in lower or "lipophilicity" in lower:
        return "logd"
    if "reactant used in the synthesis" in lower or "correct reactant" in lower or "which reactant" in lower:
        return "reactant_selection"
    if "product" in lower and ("synthesis" in lower or "reaction" in lower):
        return "product_selection"
    return "generic_science_mcq"


def _parse_numeric_science_options(question_text):
    parsed = {}
    for option, value_text in _OPTION_NUMERIC_RE.findall(question_text or ""):
        parsed[option.upper()] = float(value_text)
    return parsed if len(parsed) == 4 else None


def _nearest_option(option_values, numeric_value):
    ranked = sorted(
        ((option, abs(value - numeric_value)) for option, value in option_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    if not ranked:
        return None
    return ranked[0][0]


def _extract_keyword_numbers(reasoning_text, keywords, require_decimal=False):
    candidates = []
    for raw_line in reasoning_text.splitlines():
        line = _LIST_MARKER_RE.sub("", raw_line.strip())
        if not line:
            continue
        lower = line.lower()
        if not any(keyword in lower for keyword in keywords):
            continue
        for match in _NUMBER_RE.findall(line):
            if require_decimal and "." not in match:
                continue
            value = float(match)
            if value >= 0:
                candidates.append(value)
    return candidates


def _extract_summary_numbers(reasoning_text, option_values):
    candidates = []
    option_value_set = {int(round(float(value))) for value in option_values.values()}
    for raw_line in reasoning_text.splitlines():
        line = _LIST_MARKER_RE.sub("", raw_line.strip())
        if not line:
            continue
        lower = line.lower()
        if not any(hint in lower for hint in _SUMMARY_LINE_HINTS):
            continue
        for match in _NUMBER_RE.findall(line):
            value = int(round(float(match)))
            if value in option_value_set and value >= 0:
                candidates.append(value)
    return candidates


def _infer_counting_reasoning_option(bucket, reasoning_text, option_values):
    if not reasoning_text or not option_values:
        return None

    keywords = _KEYWORD_LINES_BY_BUCKET.get(bucket, ("count", "total"))
    if bucket == "molar_weight":
        candidates = _extract_keyword_numbers(reasoning_text, keywords, require_decimal=True)
        if not candidates:
            all_values = [float(match) for match in _NUMBER_RE.findall(reasoning_text) if "." in match]
            candidates = [value for value in all_values if value >= 20.0]
        if not candidates:
            return None
        numeric_value = candidates[-1]
        return _nearest_option(option_values, numeric_value)

    candidates = _extract_keyword_numbers(reasoning_text, keywords, require_decimal=False)
    if not candidates:
        candidates = _extract_summary_numbers(reasoning_text, option_values)
    if not candidates:
        return None

    numeric_value = int(round(float(candidates[-1])))
    for option, option_value in option_values.items():
        if int(round(option_value)) == numeric_value:
            return option
    return None


def _extract_actions(text):
    return [item.strip() for item in _ACTION_RE.findall(text)]


def _extract_action_inputs(text):
    combined = {}
    parsed_any = False
    for block in _ACTION_INPUT_RE.findall(text):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            combined.update(parsed)
            parsed_any = True
    return combined, parsed_any


def tooluse_verifier_score(completion_text, golden_answer):
    if not golden_answer:
        return 1.0

    predicted_actions = _extract_actions(completion_text)
    predicted_inputs, parsed_any_input = _extract_action_inputs(completion_text)

    target_actions = [item.get("Action", "").strip() for item in golden_answer]
    target_inputs = {}
    for item in golden_answer:
        raw_input = item.get("Action_Input", "{}")
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            target_inputs.update(parsed)

    if not predicted_actions:
        return 0.2

    if Counter(predicted_actions) != Counter(target_actions):
        return 0.5

    if target_inputs and not parsed_any_input:
        return 0.2

    normalized_pred = _normalize_obj(predicted_inputs)
    normalized_target = _normalize_obj(target_inputs)

    if normalized_pred == normalized_target:
        return 2.0

    if isinstance(normalized_pred, dict) and isinstance(normalized_target, dict):
        shared_keys = set(normalized_pred) & set(normalized_target)
        if shared_keys and all(normalized_pred[key] == normalized_target[key] for key in shared_keys):
            return 1.5

    return 1.0


def science_verifier_score(completion_text, gold_answer, question_text=None, question_bucket=None):
    predicted_answer = extract_science_answer(completion_text)
    if predicted_answer is None:
        return 0.2

    if gold_answer is None:
        return 1.0

    predicted_answer = predicted_answer.strip().upper()
    gold_answer = gold_answer.strip().upper()
    answer_correct = predicted_answer == gold_answer
    bucket = question_bucket or _infer_science_bucket(question_text)

    if bucket not in _COUNTING_BUCKETS or not question_text:
        return 2.0 if answer_correct else 0.5

    option_values = _parse_numeric_science_options(question_text)
    if not option_values:
        return 2.0 if answer_correct else 0.5

    reasoning_text = _extract_reasoning(completion_text)
    reasoning_option = _infer_counting_reasoning_option(bucket, reasoning_text, option_values)

    if answer_correct:
        if reasoning_option is None:
            return 1.5
        if reasoning_option == predicted_answer:
            return 2.0
        return 1.0

    if reasoning_option is None:
        return 0.5
    if reasoning_option == predicted_answer:
        return 0.4
    return 0.2
