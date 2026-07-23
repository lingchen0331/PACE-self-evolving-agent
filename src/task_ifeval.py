import copy
import json
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from wrap import wrap_solver

threshold = 0.50
full_eval_improvement_epsilon = 0.03
last_test_acc = 0.0
last_full_eval_valid_strict = None
force_first_full_eval = True

DEFAULT_PROMPT_CONFIG = {
    "role": "instruction-following expert",
    "requirements": (
        "1. Read the user's prompt carefully and follow EVERY constraint.\n"
        "2. Your entire reply IS the response — do not wrap it in JSON or any other container.\n"
        "3. The response MUST satisfy all formatting, keyword, length, and style constraints in the prompt."
    ).strip(),
    "temperature": 0.2,
}

IFEVAL_PROMPT_MUTATIONS = [
    {
        "requirements_suffix": (
            "4. Before writing, list every constraint you must satisfy, then write the response."
        )
    },
    {
        "requirements_suffix": (
            "4. Double-check each constraint after drafting your response."
        )
    },
    {
        "requirements_suffix": (
            "4. Pay special attention to punctuation, case, and word-count constraints."
        )
    },
    {"temperature": 0.1},
    {"temperature": 0.0},
    {
        "role": "meticulous instruction-following assistant that never misses a constraint",
    },
]

# ---------------------------------------------------------------------------
# Instruction verification functions
# Each takes (response_text, **kwargs) and returns bool
# ---------------------------------------------------------------------------


def _check_punctuation_no_comma(response, **kwargs):
    return "," not in response


def _check_startend_end_checker(response, **kwargs):
    end_phrase = kwargs.get("end_phrase", "")
    return response.rstrip().endswith(end_phrase)


def _check_startend_quotation(response, **kwargs):
    stripped = response.strip()
    return stripped.startswith('"') and stripped.endswith('"')


def _check_change_case_english_lowercase(response, **kwargs):
    return response == response.lower()


def _check_change_case_english_capital(response, **kwargs):
    return response == response.upper()


def _check_change_case_capital_word_frequency(response, **kwargs):
    capital_relation = kwargs.get("capital_relation", "at least")
    capital_frequency = int(kwargs.get("capital_frequency", 1))
    words = response.split()
    capital_count = sum(1 for w in words if w.isupper() and w.isalpha())
    return _check_relation(capital_count, capital_relation, capital_frequency)


def _check_relation(value, relation, threshold_val):
    if relation in ("at least", "at_least"):
        return value >= threshold_val
    elif relation in ("at most", "at_most"):
        return value <= threshold_val
    elif relation in ("less than", "less_than"):
        return value < threshold_val
    elif relation in ("more than", "more_than"):
        return value > threshold_val
    elif relation == "exactly":
        return value == threshold_val
    return value >= threshold_val


def _check_length_constraints_number_words(response, **kwargs):
    relation = kwargs.get("relation", "at least")
    num_words = int(kwargs.get("num_words", 1))
    word_count = len(response.split())
    return _check_relation(word_count, relation, num_words)


def _check_length_constraints_number_sentences(response, **kwargs):
    relation = kwargs.get("relation", "at least")
    num_sentences = int(kwargs.get("num_sentences", 1))
    sentences = re.split(r'[.!?]+', response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return _check_relation(len(sentences), relation, num_sentences)


def _check_length_constraints_number_paragraphs(response, **kwargs):
    num_paragraphs = int(kwargs.get("num_paragraphs", 1))
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    return len(paragraphs) >= num_paragraphs


def _check_length_constraints_nth_paragraph_first_word(response, **kwargs):
    first_word = kwargs.get("first_word", "").lower()
    num_paragraphs = int(kwargs.get("num_paragraphs", 1))
    nth_paragraph = int(kwargs.get("nth_paragraph", 1))
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) < num_paragraphs:
        return False
    idx = nth_paragraph - 1
    if idx < 0 or idx >= len(paragraphs):
        return False
    words = paragraphs[idx].split()
    if not words:
        return False
    return words[0].lower().strip("*#_") == first_word


def _check_detectable_format_number_bullet_lists(response, **kwargs):
    num_bullets = int(kwargs.get("num_bullets", 1))
    bullet_count = len(re.findall(r'^\s*[\*\-•]\s', response, re.MULTILINE))
    return bullet_count >= num_bullets


def _check_detectable_format_number_highlighted_sections(response, **kwargs):
    num_highlights = int(kwargs.get("num_highlights", 1))
    highlights = re.findall(r'\*[^*\n]+\*', response)
    return len(highlights) >= num_highlights


def _check_detectable_format_title(response, **kwargs):
    return bool(re.search(r'<<[^>]+>>', response))


def _check_detectable_format_json_format(response, **kwargs):
    try:
        json.loads(response.strip())
        return True
    except Exception:
        # Try to find JSON block in response
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return True
            except Exception:
                pass
        # Try to find any JSON object
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                json.loads(match.group(0))
                return True
            except Exception:
                pass
    return False


def _check_detectable_format_multiple_sections(response, **kwargs):
    section_splitter = kwargs.get("section_spliter", "SECTION")
    num_sections = int(kwargs.get("num_sections", 1))
    pattern = re.compile(re.escape(section_splitter), re.IGNORECASE)
    parts = pattern.split(response)
    return len(parts) >= num_sections


def _check_detectable_format_constrained_response(response, **kwargs):
    stripped = response.strip().lower()
    return stripped in ("my answer is yes.", "my answer is no.",
                        "my answer is maybe.", "my answer is yes",
                        "my answer is no", "my answer is maybe")


def _check_detectable_content_number_placeholders(response, **kwargs):
    num_placeholders = int(kwargs.get("num_placeholders", 1))
    placeholders = re.findall(r'\[.*?\]', response)
    return len(placeholders) >= num_placeholders


def _check_detectable_content_postscript(response, **kwargs):
    postscript_marker = kwargs.get("postscript_marker", "P.S.")
    return postscript_marker in response


def _check_keywords_existence(response, **kwargs):
    keywords = kwargs.get("keywords", [])
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


def _check_keywords_forbidden_words(response, **kwargs):
    forbidden_words = kwargs.get("forbidden_words", [])
    response_lower = response.lower()
    return all(fw.lower() not in response_lower for fw in forbidden_words)


def _check_keywords_frequency(response, **kwargs):
    relation = kwargs.get("relation", "at least")
    keyword = kwargs.get("keyword", "")
    frequency = int(kwargs.get("frequency", 1))
    count = response.lower().count(keyword.lower())
    return _check_relation(count, relation, frequency)


def _check_keywords_letter_frequency(response, **kwargs):
    let_relation = kwargs.get("let_relation", "at least")
    letter = kwargs.get("letter", "")
    let_frequency = int(kwargs.get("let_frequency", 1))
    count = response.lower().count(letter.lower())
    return _check_relation(count, let_relation, let_frequency)


def _check_combination_repeat_prompt(response, **kwargs):
    prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
    if not prompt_to_repeat:
        return True
    return prompt_to_repeat in response


def _check_combination_two_responses(response, **kwargs):
    separators = ["******", "---", "***", "==="]
    for sep in separators:
        if sep in response:
            parts = response.split(sep)
            non_empty = [p.strip() for p in parts if p.strip()]
            if len(non_empty) >= 2:
                return True
    # Also check for numbered responses
    if re.search(r'Response\s*1', response, re.IGNORECASE) and \
       re.search(r'Response\s*2', response, re.IGNORECASE):
        return True
    return False


def _check_language_response_language(response, **kwargs):
    # Lightweight heuristic: just check it's non-empty.
    # Full language detection would require langdetect/fasttext.
    # We accept this as a best-effort check.
    return len(response.strip()) > 0


# ---------------------------------------------------------------------------
# Registry mapping instruction_id -> checker function
# ---------------------------------------------------------------------------

INSTRUCTION_CHECKERS = {
    "punctuation:no_comma": _check_punctuation_no_comma,
    "startend:end_checker": _check_startend_end_checker,
    "startend:quotation": _check_startend_quotation,
    "change_case:english_lowercase": _check_change_case_english_lowercase,
    "change_case:english_capital": _check_change_case_english_capital,
    "change_case:capital_word_frequency": _check_change_case_capital_word_frequency,
    "length_constraints:number_words": _check_length_constraints_number_words,
    "length_constraints:number_sentences": _check_length_constraints_number_sentences,
    "length_constraints:number_paragraphs": _check_length_constraints_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _check_length_constraints_nth_paragraph_first_word,
    "detectable_format:number_bullet_lists": _check_detectable_format_number_bullet_lists,
    "detectable_format:number_highlighted_sections": _check_detectable_format_number_highlighted_sections,
    "detectable_format:title": _check_detectable_format_title,
    "detectable_format:json_format": _check_detectable_format_json_format,
    "detectable_format:multiple_sections": _check_detectable_format_multiple_sections,
    "detectable_format:constrained_response": _check_detectable_format_constrained_response,
    "detectable_content:number_placeholders": _check_detectable_content_number_placeholders,
    "detectable_content:postscript": _check_detectable_content_postscript,
    "keywords:existence": _check_keywords_existence,
    "keywords:forbidden_words": _check_keywords_forbidden_words,
    "keywords:frequency": _check_keywords_frequency,
    "keywords:letter_frequency": _check_keywords_letter_frequency,
    "combination:repeat_prompt": _check_combination_repeat_prompt,
    "combination:two_responses": _check_combination_two_responses,
    "language:response_language": _check_language_response_language,
}


def verify_instruction(response_text, instruction_id, kwargs_dict):
    """Check a single instruction constraint against the response."""
    checker = INSTRUCTION_CHECKERS.get(instruction_id)
    if checker is None:
        return True  # unknown instruction, skip
    try:
        return checker(response_text, **kwargs_dict)
    except Exception:
        return False


def verify_all_instructions(response_text, instruction_id_list, kwargs_list):
    """Return (strict_acc, loose_acc, per_instruction_results).
    strict_acc = 1 if ALL instructions pass, else 0
    loose_acc  = fraction of instructions that pass
    """
    results = []
    for iid, kw in zip(instruction_id_list, kwargs_list):
        results.append(verify_instruction(response_text, iid, kw))
    strict = 1 if all(results) else 0
    loose = sum(results) / len(results) if results else 0.0
    return strict, loose, results


# ---------------------------------------------------------------------------
# Solver & data loading
# ---------------------------------------------------------------------------

def solver(agent, task: str):
    prompt_cfg = normalize_prompt_config(
        getattr(agent, "prompt_config", DEFAULT_PROMPT_CONFIG)
    )
    system_prompt = (
        f"You are a helpful {prompt_cfg['role']}.\n"
        f"Requirements:\n{prompt_cfg['requirements']}"
    ).strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    responses = agent.action_call_llm(
        model=getattr(agent, "default_infer_model", "gpt-4.1-mini"),
        messages=messages,
        temperature=prompt_cfg["temperature"],
        n=1,
        response_format="text",
    )
    response_text = str(getattr(responses[0], "content", responses[0]) if not isinstance(responses[0], str) else responses[0]).strip()
    return {"response": response_text}


def _load_ifeval_examples(split="train"):
    if split == "train":
        path = "datasets/ifeval/ifeval_input_data_train.jsonl"
    else:
        path = "datasets/ifeval/ifeval_input_data_test.jsonl"
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    random.seed(0)
    random.shuffle(examples)
    return examples


def _single_example_eval(agent, prompt_text, instruction_id_list, kwargs_list, prompt_config):
    try:
        system_prompt = (
            f"You are a helpful {prompt_config['role']}.\n"
            f"Requirements:\n{prompt_config['requirements']}"
        ).strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        responses = agent.action_call_llm(
            model=getattr(agent, "default_infer_model", "gpt-4.1-mini"),
            messages=messages,
            temperature=prompt_config["temperature"],
            n=1,
            response_format="text",
        )
        raw = responses[0]
        response_text = str(getattr(raw, "content", raw) if not isinstance(raw, str) else raw).strip()
        strict, loose, per_instr = verify_all_instructions(
            response_text, instruction_id_list, kwargs_list
        )
        return {
            "ok": True,
            "response": response_text,
            "strict": strict,
            "loose": loose,
            "per_instruction": per_instr,
            "cost": len(response_text),
            "raw": response_text,
        }
    except Exception as e:
        return {
            "ok": False,
            "response": "",
            "strict": 0,
            "loose": 0.0,
            "per_instruction": [],
            "cost": 10000,
            "raw": repr(e),
        }


def _evaluate_prompt_config(agent, prompt_config, examples, max_workers=16):
    prompt_config = normalize_prompt_config(prompt_config)
    effective_max = getattr(agent, "max_workers", max_workers)
    worker = max(1, min(effective_max, len(examples)))

    with ThreadPoolExecutor(max_workers=worker) as executor:
        outputs = list(
            executor.map(
                lambda ex: _single_example_eval(
                    agent, ex["prompt"],
                    ex["instruction_id_list"], ex["kwargs"],
                    prompt_config,
                ),
                examples,
            )
        )

    strict_list = []
    loose_list = []
    total_cost = 0
    failures = []
    for idx, out in enumerate(outputs):
        strict_list.append(out["strict"])
        loose_list.append(out["loose"])
        total_cost += out["cost"]
        if out["strict"] == 0 and len(failures) < 6:
            failures.append({
                "prompt": examples[idx]["prompt"][:600],
                "instruction_ids": examples[idx]["instruction_id_list"],
                "per_instruction": out.get("per_instruction", []),
                "response_preview": out.get("response", "")[:400],
            })

    strict_acc = float(sum(strict_list) / len(strict_list)) if strict_list else 0.0
    loose_acc = float(sum(loose_list) / len(loose_list)) if loose_list else 0.0
    avg_cost = float(total_cost / len(outputs)) if outputs else 0.0
    return {
        "accuracy": strict_acc,
        "loose_accuracy": loose_acc,
        "avg_cost": avg_cost,
        "failures": failures,
        "num_samples": len(outputs),
        "prompt_config": prompt_config,
    }


def _screen_prompt_candidates(agent, candidates, train_examples, candidates_per_iter):
    if len(candidates) <= 3 or len(train_examples) <= 16:
        return list(candidates)

    screened = []
    for cand in candidates:
        screened_eval = _evaluate_prompt_config(agent, cand, train_examples[:16])
        screened.append(screened_eval)

    screened.sort(key=lambda x: (-x["accuracy"], x["avg_cost"]))
    keep_count = min(len(screened), max(3, min(6, int(candidates_per_iter) // 2 + 1)))
    return [entry["prompt_config"] for entry in screened[:keep_count]]


# ---------------------------------------------------------------------------
# Prompt optimization (mirrors MMLU pattern)
# ---------------------------------------------------------------------------

def normalize_prompt_config(prompt_config):
    merged = copy.deepcopy(DEFAULT_PROMPT_CONFIG)
    if isinstance(prompt_config, dict):
        for key in ["role", "requirements", "temperature"]:
            if key in prompt_config and prompt_config[key] is not None:
                merged[key] = prompt_config[key]
    try:
        merged["temperature"] = float(merged["temperature"])
    except Exception:
        merged["temperature"] = DEFAULT_PROMPT_CONFIG["temperature"]
    merged["temperature"] = max(0.0, min(1.0, merged["temperature"]))
    return merged


def _build_mutated_prompt_config(base_prompt, mutation, revise=False):
    cand = normalize_prompt_config(base_prompt)
    if revise:
        cand["requirements"] = DEFAULT_PROMPT_CONFIG["requirements"]

    if "role" in mutation:
        cand["role"] = mutation["role"]
    if "temperature" in mutation:
        cand["temperature"] = mutation["temperature"]
    if "requirements" in mutation:
        cand["requirements"] = str(mutation["requirements"]).strip()
    elif "requirements_suffix" in mutation:
        cand["requirements"] = (
            cand["requirements"].strip() + "\n" + mutation["requirements_suffix"].strip()
        )

    return normalize_prompt_config(cand)


def _generate_handcrafted_candidates(base_prompt, candidates_per_iter):
    candidates = [normalize_prompt_config(base_prompt)]
    seen = {json.dumps(candidates[0], sort_keys=True)}
    for mutation in IFEVAL_PROMPT_MUTATIONS:
        for revise in (False, True):
            if revise and "requirements_suffix" not in mutation and "requirements" not in mutation:
                continue
            cand = _build_mutated_prompt_config(base_prompt, mutation, revise=revise)
            key = json.dumps(cand, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
            if len(candidates) >= candidates_per_iter:
                return candidates
    return candidates


def _generate_reflective_candidates(agent, base_prompt, failures, max_new=2):
    if not failures or max_new <= 0:
        return []

    failure_preview = []
    for failure in failures[:3]:
        failure_preview.append({
            "prompt": failure["prompt"][:400],
            "instruction_ids": failure["instruction_ids"],
            "per_instruction": failure["per_instruction"],
        })

    reflection_messages = [
        {
            "role": "user",
            "content": (
                "You are optimizing an IFEval prompt for a small model. "
                "The model must follow every instruction constraint exactly. "
                "Given current prompt config and failed examples, propose diverse prompt improvements.\n"
                "Return full revised prompt configs, not suffixes or patches. "
                "If you change requirements, provide the complete replacement requirements string.\n"
                f"Current prompt config:\n{json.dumps(base_prompt, ensure_ascii=True, indent=2)}\n"
                f"Failed examples:\n{json.dumps(failure_preview, ensure_ascii=True, indent=2)}"
            ),
        }
    ]

    try:
        response = agent.action_call_json_format_llm(
            model=getattr(agent, "optimization_model",
                          getattr(agent, "default_infer_model", "gpt-4.1-mini")),
            messages=reflection_messages,
            temperature=0.7,
            num_of_response=1,
            role="prompt optimizer",
            return_dict_keys=["candidates"],
            requirements=(
                "Return key 'candidates' as a list of dictionaries. "
                "Each dict can include role, requirements, and temperature. "
                "Each candidate must be a standalone revised prompt config, not an incremental diff. "
                "Provide at most 4 candidates."
            ),
        )
    except Exception:
        return []

    payload = response[0] if response else {}
    raw_candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    if isinstance(raw_candidates, dict):
        raw_candidates = [raw_candidates]

    normalized = []
    for cand in raw_candidates:
        if not isinstance(cand, dict):
            continue
        merged = normalize_prompt_config({**base_prompt, **cand})
        normalized.append(merged)
        if len(normalized) >= max_new:
            break
    return normalized


def _dominates(a, b):
    return (
        a["accuracy"] >= b["accuracy"]
        and a["avg_cost"] <= b["avg_cost"]
        and (a["accuracy"] > b["accuracy"] or a["avg_cost"] < b["avg_cost"])
    )


def _update_pareto_front(pareto_front, new_candidates, max_size=6):
    all_candidates = pareto_front + new_candidates
    filtered = []
    for cand in all_candidates:
        dominated = False
        for other in all_candidates:
            if other is cand:
                continue
            if _dominates(other, cand):
                dominated = True
                break
        if not dominated:
            filtered.append(cand)
    filtered.sort(key=lambda x: (-x["accuracy"], x["avg_cost"]))
    return filtered[:max_size]


def optimize_prompt_config(
    agent,
    current_prompt_config,
    iterations=2,
    train_size=24,
    valid_size=24,
    candidates_per_iter=5,
    seed=7,
    selected_train_examples=None,
    selected_valid_examples=None,
):
    examples = _load_ifeval_examples("train")
    random.seed(seed)
    in_domain = examples[:128]
    random.shuffle(in_domain)

    if selected_train_examples is not None:
        train_examples = list(selected_train_examples)[: max(1, int(train_size))]
    else:
        train_examples = in_domain[: max(8, train_size)]
    if selected_valid_examples is not None:
        valid_examples = list(selected_valid_examples)[: max(1, int(valid_size))]
    else:
        valid_examples = in_domain[max(8, train_size): max(8, train_size) + max(8, valid_size)]
        if not valid_examples:
            valid_examples = in_domain[-max(8, valid_size):]

    current_prompt_config = normalize_prompt_config(current_prompt_config)
    print(f"[inner-loop] Starting prompt optimization: {iterations} iterations, "
          f"{len(train_examples)} train / {len(valid_examples)} valid samples, "
          f"{candidates_per_iter} candidates/iter", flush=True)
    evaluated = _evaluate_prompt_config(agent, current_prompt_config, train_examples)
    print(f"[inner-loop] Initial train accuracy: {evaluated['accuracy']:.4f}", flush=True)
    pareto_front = [evaluated]
    history = [
        {
            "stage": "initial",
            "accuracy": evaluated["accuracy"],
            "avg_cost": evaluated["avg_cost"],
            "prompt_config": evaluated["prompt_config"],
        }
    ]

    for iter_idx in range(max(0, int(iterations))):
        iter_start = time.time()
        best_now = max(pareto_front, key=lambda x: (x["accuracy"], -x["avg_cost"]))
        handcrafted = _generate_handcrafted_candidates(
            best_now["prompt_config"],
            candidates_per_iter=max(2, int(candidates_per_iter)),
        )
        reflective = _generate_reflective_candidates(
            agent,
            best_now["prompt_config"],
            best_now["failures"],
            max_new=max(0, int(candidates_per_iter) - len(handcrafted) + 1),
        )

        raw_candidates = handcrafted + reflective
        dedup = []
        seen = set()
        for cand in raw_candidates:
            key = json.dumps(cand, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(cand)

        print(f"[inner-loop] Iter {iter_idx+1}/{iterations}: "
              f"{len(handcrafted)} handcrafted + {len(reflective)} reflective → "
              f"{len(dedup)} unique candidates", flush=True)

        eval_results = []
        narrowed_candidates = _screen_prompt_candidates(
            agent,
            dedup,
            train_examples,
            candidates_per_iter,
        )
        print(f"[inner-loop] Iter {iter_idx+1}/{iterations}: "
              f"screening kept {len(narrowed_candidates)}/{len(dedup)} candidates", flush=True)
        for cand_idx, cand in enumerate(narrowed_candidates):
            cand_eval = _evaluate_prompt_config(agent, cand, train_examples)
            eval_results.append(cand_eval)
            print(f"[inner-loop] Iter {iter_idx+1}/{iterations}: "
                  f"candidate {cand_idx+1}/{len(narrowed_candidates)} → "
                  f"acc={cand_eval['accuracy']:.4f}", flush=True)

        pareto_front = _update_pareto_front(pareto_front, eval_results)
        iter_best = max(pareto_front, key=lambda x: (x["accuracy"], -x["avg_cost"]))
        elapsed = time.time() - iter_start
        print(f"[inner-loop] Iter {iter_idx+1}/{iterations} done in {elapsed:.1f}s: "
              f"best_acc={iter_best['accuracy']:.4f}, pareto_size={len(pareto_front)}", flush=True)
        history.append({
            "stage": "iteration",
            "accuracy": iter_best["accuracy"],
            "avg_cost": iter_best["avg_cost"],
            "prompt_config": iter_best["prompt_config"],
        })

    print(f"[inner-loop] Validation phase: evaluating baseline + {len(pareto_front)} pareto candidates "
          f"on {len(valid_examples)} valid samples", flush=True)
    initial_valid_eval = _evaluate_prompt_config(agent, current_prompt_config, valid_examples)
    print(f"[inner-loop] Baseline valid accuracy: {initial_valid_eval['accuracy']:.4f}", flush=True)

    valid_scored = []
    for vi, cand in enumerate(pareto_front):
        valid_eval = _evaluate_prompt_config(agent, cand["prompt_config"], valid_examples)
        print(f"[inner-loop] Valid candidate {vi+1}/{len(pareto_front)}: "
              f"acc={valid_eval['accuracy']:.4f}", flush=True)
        valid_scored.append({
            "train_accuracy": cand["accuracy"],
            "train_avg_cost": cand["avg_cost"],
            "valid_accuracy": valid_eval["accuracy"],
            "valid_avg_cost": valid_eval["avg_cost"],
            "prompt_config": cand["prompt_config"],
        })

    valid_scored.sort(key=lambda x: (-x["valid_accuracy"], x["valid_avg_cost"], -x["train_accuracy"]))
    best = valid_scored[0]
    delta_u_p = float(best["valid_accuracy"] - initial_valid_eval["accuracy"])

    report = {
        "best_prompt_config": best["prompt_config"],
        "best_valid_accuracy": best["valid_accuracy"],
        "best_valid_avg_cost": best["valid_avg_cost"],
        "initial_valid_accuracy": initial_valid_eval["accuracy"],
        "initial_valid_avg_cost": initial_valid_eval["avg_cost"],
        "delta_u_p": delta_u_p,
        "history": history,
        "candidates_considered": len(valid_scored),
        "train_samples": len(train_examples),
        "valid_samples": len(valid_examples),
    }

    report_text = (
        f"Inner Prompt Optimization Finished.\\n"
        f"Train samples: {report['train_samples']}, Valid samples: {report['valid_samples']}\\n"
        f"Candidates on Pareto front: {report['candidates_considered']}\\n"
        f"Initial valid accuracy: {report['initial_valid_accuracy']:.4f}\\n"
        f"Best valid accuracy: {report['best_valid_accuracy']:.4f}, "
        f"best valid avg_cost: {report['best_valid_avg_cost']:.1f}\\n"
        f"Estimated Delta_U_P: {report['delta_u_p']:.4f}\\n"
        f"Best prompt config: {json.dumps(report['best_prompt_config'], ensure_ascii=True)}"
    )

    return report, report_text


# ---------------------------------------------------------------------------
# Real evaluate & Task class
# ---------------------------------------------------------------------------

def real_evaluate(solver, max_workers=48):
    examples = _load_ifeval_examples("test")
    max_workers = min(len(examples), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(wrap_solver(solver), [ex["prompt"] for ex in examples]),
            total=len(examples),
        ))

    strict_list = []
    loose_list = []
    info_list = []
    for idx, res in enumerate(results):
        ex = examples[idx]
        try:
            response_text = str(res.get("response", res) if isinstance(res, dict) else res)
            strict, loose, per_instr = verify_all_instructions(
                response_text,
                ex["instruction_id_list"],
                ex["kwargs"],
            )
        except Exception as e:
            info_list.append(f"Sample {idx}:\n{repr(e)}\nModel Output: {res}\n")
            strict_list.append(0)
            loose_list.append(0.0)
            continue

        strict_list.append(strict)
        loose_list.append(loose)

        failed_ids = [
            iid for iid, passed in zip(ex["instruction_id_list"], per_instr) if not passed
        ]
        info_list.append(
            f"Sample {idx} (key={ex['key']}):\n"
            f"Prompt: {ex['prompt'][:200]}...\n"
            f"Response: {response_text[:300]}...\n"
            f"Strict: {strict}  Loose: {loose:.2f}  "
            f"Failed: {failed_ids}\n"
        )

    strict_acc = sum(strict_list) / len(strict_list) if strict_list else 0.0
    loose_acc = sum(loose_list) / len(loose_list) if loose_list else 0.0
    interval = bootstrap_confidence_interval(strict_list)
    if strict_acc > last_test_acc:
        with open(f"results/ifeval_{round(strict_acc, 4)}.txt", "w") as f:
            f.writelines(
                [f"Strict: {strict_acc:.4f}  Loose: {loose_acc:.4f}\n", interval + "\n"]
                + info_list
            )
    return strict_acc


class IFEVAL_Task:
    def evaluate(self, solver, examples_override=None, max_workers=48):
        global force_first_full_eval
        global last_full_eval_valid_strict
        global last_test_acc
        if examples_override is not None:
            examples = list(examples_override)
        else:
            examples = _load_ifeval_examples("train")[:128]
            random.seed(time.time())
            random.shuffle(examples)
            examples = examples[:20]

        workers = min(len(examples), max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(solver, [ex["prompt"] for ex in examples]),
                total=len(examples),
            ))

        strict_list = []
        loose_list = []
        info_list = []
        for idx, res in enumerate(results):
            ex = examples[idx]
            try:
                response_text = str(res.get("response", res) if isinstance(res, dict) else res)
                strict, loose, per_instr = verify_all_instructions(
                    response_text,
                    ex["instruction_id_list"],
                    ex["kwargs"],
                )
            except Exception as e:
                info_list.append(f"Valid Sample {idx}:\n{repr(e)}\nModel Output: {res}\n")
                strict_list.append(0)
                loose_list.append(0.0)
                continue

            strict_list.append(strict)
            loose_list.append(loose)

            failed_ids = [
                iid for iid, passed in zip(ex["instruction_id_list"], per_instr) if not passed
            ]
            info_list.append(
                f"Valid Sample {idx} (key={ex['key']}):\n"
                f"Prompt: {ex['prompt'][:200]}...\n"
                f"Response: {response_text[:300]}...\n"
                f"Strict: {strict}  Loose: {loose:.2f}  "
                f"Failed: {failed_ids}\n"
            )

        valid_strict = sum(strict_list) / len(strict_list) if strict_list else 0.0
        valid_loose = sum(loose_list) / len(loose_list) if loose_list else 0.0
        print(f"Strict Acc: {valid_strict:.4f}  Loose Acc: {valid_loose:.4f}")

        should_run_test = force_first_full_eval
        if not should_run_test and valid_strict >= threshold:
            if last_full_eval_valid_strict is None:
                should_run_test = True
            elif valid_strict >= last_full_eval_valid_strict + full_eval_improvement_epsilon:
                should_run_test = True
        if should_run_test:
            test_acc = real_evaluate(solver, max_workers=max_workers)
            last_test_acc = test_acc
            last_full_eval_valid_strict = valid_strict
            gate_note = ""
            if force_first_full_eval and valid_strict < threshold:
                gate_note = (
                    f"Forced first full evaluation despite valid strict accuracy below {threshold}.\n"
                )
            force_first_full_eval = False
            feedback = (
                f"Valid Strict Accuracy: {valid_strict}\nValid Loose Accuracy: {valid_loose}\n"
                + gate_note
                + 
                f"Test Strict Accuracy: {test_acc}\n"
                + "Evaluation Info:\n"
                + "\n".join(info_list)
            )
        else:
            test_acc = last_test_acc
            feedback = (
                f"Valid Strict Accuracy: {valid_strict}\nValid Loose Accuracy: {valid_loose}\n"
                f"Skipped full test evaluation; threshold={threshold}, last_full_eval_valid={last_full_eval_valid_strict}, epsilon={full_eval_improvement_epsilon}.\n"
                f"Cached Test Strict Accuracy: {test_acc}\n"
                + "Evaluation Info:\n"
                + "\n".join(info_list)
            )
        return feedback, test_acc


def bootstrap_confidence_interval(data, confidence_level=0.95):
    n = len(data)
    if n == 0:
        return "95% Confidence Interval: (0.0%, 0.0%), Median: 0.0%"

    mean = sum(data) / n
    z = 1.96 if abs(confidence_level - 0.95) < 1e-9 else 1.96
    denom = 1.0 + (z * z) / n
    center = (mean + (z * z) / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((mean * (1.0 - mean) / n) + ((z * z) / (4.0 * n * n)))
        / denom
    )
    ci_lower_percent = max(0.0, center - margin) * 100
    ci_upper_percent = min(1.0, center + margin) * 100
    median_percent = mean * 100

    print(
        f"95% Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), "
        f"Median: {median_percent:.1f}%"
    )
    return (
        f"95% Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), "
        f"Median: {median_percent:.1f}%"
    )
