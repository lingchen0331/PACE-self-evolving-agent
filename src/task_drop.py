# https://github.com/openai/simple-evals/blob/main/drop_eval.py
"""
DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, Matt Gardner
https://arxiv.org/abs/1903.00161
"""

import copy
import gzip
import json
import math
import random
import re
import string
import time
from typing import Any, Dict, List, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from wrap import wrap_solver

threshold = 0.70
last_test_acc = 0.0
force_first_full_eval = True

DEFAULT_PROMPT_CONFIG = {
    "role": "reading comprehension expert",
    "requirements": (
        "1. Please explain step by step.\n"
        "2. Directly answer the question.\n"
        "3. The answer MUST be a concise string."
    ).strip(),
    "temperature": 0.2,
}

DROP_PROMPT_MUTATIONS = [
    {
        "requirements_suffix": (
            "4. First identify the relevant numbers or facts in the passage, then compute the answer."
        )
    },
    {
        "requirements_suffix": (
            "4. Double-check any arithmetic before giving the final answer."
        )
    },
    {
        "requirements_suffix": (
            "4. If the question asks 'how many more', compute the difference explicitly."
        )
    },
    {"temperature": 0.1},
    {"temperature": 0.0},
    {
        "role": "meticulous reading comprehension and discrete reasoning specialist",
    },
]


# ---------------------------------------------------------------------------
# Solver & data loading
# ---------------------------------------------------------------------------

def solver(agent, task: str):
    prompt_cfg = normalize_prompt_config(
        getattr(agent, "prompt_config", DEFAULT_PROMPT_CONFIG)
    )
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model=getattr(agent, "default_infer_model", "gpt-4.1-mini"),
        messages=messages,
        temperature=prompt_cfg["temperature"],
        num_of_response=1,
        role=prompt_cfg["role"],
        return_dict_keys=["reasoning", "answer"],
        requirements=prompt_cfg["requirements"],
    )

    return_dict = response[0] if response else {}
    if not isinstance(return_dict, dict):
        return_dict = {"answer": str(return_dict or "")}
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict


def _coerce_solver_output(task_text, output):
    """Normalize solver output for better exact-match scoring."""
    if isinstance(output, dict):
        return_dict = dict(output)
    else:
        return_dict = {"answer": str(output or "")}

    raw_answer = str(return_dict.get("answer", "")).strip()

    # Strip trailing percent sign: "74.7%" -> "74.7"
    if raw_answer.endswith("%"):
        raw_answer = raw_answer[:-1].strip()

    # Strip commas from numbers: "331,327" -> "331327"
    if re.match(r'^[\d,]+\.?\d*$', raw_answer):
        raw_answer = raw_answer.replace(",", "")

    # Strip leading dollar/currency signs
    if raw_answer.startswith("$"):
        raw_answer = raw_answer[1:].strip()

    # Normalize trailing ".0" for integers: "6.0" -> "6"
    if re.match(r'^\d+\.0+$', raw_answer):
        raw_answer = raw_answer.split(".")[0]

    return_dict["answer"] = raw_answer
    return return_dict


def load_drop(file_path):
    if file_path.endswith(".gz"):
        with gzip.open(file_path, mode="rb") as f:
            test_samples = [json.loads(line) for line in f]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            test_samples = [json.loads(line) for line in f if line.strip()]
    few_shot_prompt = """You will be asked to read a passage and answer a question.

# Examples:
Passage: As of the census of 2000, there were 952 people, 392 households, and 241 families residing in the village. The population density was 952.9 people per square mile (367.6/km²). There were 449 housing units at an average density of 449.4 per square mile (173.4/km²). The racial makeup of the village was 96.11% White (U.S. Census), 0.95% African American (U.S. Census) or Race (United States Census), 0.11% Native American (U.S. Census), 0.11% Asian (U.S. Census), 0.21% from Race (United States Census), and 2.52% from two or more races. 1.05% of the population were Hispanics in the United States or Latino (U.S. Census) of any race.
Question: How many more people, in terms of percentage, were from two or more races compared to being solely Native American or solely Asian?
Answer: 2.3

# Your Task
---

"""
    examples = []
    for sample in test_samples:
        sample['inputs'] = few_shot_prompt + sample['context']
        sample['targets'] = sample["ref_text"].split("|")
        examples.append(sample)
    return examples


def _load_drop_examples():
    data_filename = "datasets/drop_v0_dev.jsonl"
    examples = load_drop(data_filename)
    # Reserve first and last for few-shot; shuffle the rest deterministically
    examples = examples[1:-1]
    random.seed(0)
    random.shuffle(examples)
    return examples



# ---------------------------------------------------------------------------
# DROP metric functions
# ---------------------------------------------------------------------------

def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(
        answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    ) * 100
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_drop_metrics(
        predicted: Union[str, List[str], Tuple[str, ...]],
        gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float]:
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return float(exact_match), float(f1)


def drop_metric(sample: str, reference: list[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for answer in reference:
        if answer.strip() != "":
            em, f1 = get_drop_metrics(sample, answer)
            em_scores.append(em)
            f1_scores.append(f1)
    return (max(em_scores), max(f1_scores))



# ---------------------------------------------------------------------------
# Inner-loop evaluation helpers
# ---------------------------------------------------------------------------

def _single_example_eval(agent, question, correct_answers, prompt_config):
    try:
        messages = [{"role": "user", "content": f"# Your Task:\n{question}"}]
        response = agent.action_call_json_format_llm(
            model=getattr(agent, "default_infer_model", "gpt-4.1-mini"),
            messages=messages,
            temperature=prompt_config["temperature"],
            num_of_response=1,
            role=prompt_config["role"],
            return_dict_keys=["reasoning", "answer"],
            requirements=prompt_config["requirements"],
        )
        output = response[0] if response else {}
        if not isinstance(output, dict):
            output = {"answer": str(output or "")}
        extracted_answer = str(output.get("answer", ""))
        reasoning = str(output.get("reasoning", ""))
        em_score, f1_score = drop_metric(extracted_answer, correct_answers)
        return {
            "ok": True,
            "reasoning": reasoning,
            "answer": extracted_answer,
            "f1": f1_score,
            "em": em_score,
            "cost": len(reasoning) + len(extracted_answer),
            "raw": output,
        }
    except Exception as e:
        return {
            "ok": False,
            "reasoning": "",
            "answer": "",
            "f1": 0.0,
            "em": 0.0,
            "cost": 10000,
            "raw": repr(e),
        }


def _evaluate_prompt_config(agent, prompt_config, examples, max_workers=16):
    questions = [ex["inputs"] for ex in examples]
    answers = [ex["targets"] for ex in examples]

    prompt_config = normalize_prompt_config(prompt_config)
    worker = max(1, min(max_workers, len(questions)))

    with ThreadPoolExecutor(max_workers=worker) as executor:
        outputs = list(
            executor.map(
                lambda idx: _single_example_eval(agent, questions[idx], answers[idx], prompt_config),
                range(len(questions)),
            )
        )

    f1_list = []
    total_cost = 0
    failures = []
    for idx, out in enumerate(outputs):
        f1_list.append(out["em"])  # exact match: 0.0 or 1.0
        total_cost += out["cost"]
        if out["em"] < 1.0 and len(failures) < 6:
            failures.append({
                "question": questions[idx][:600],
                "pred": out.get("answer", ""),
                "gold": answers[idx],
                "raw": out.get("raw", ""),
            })

    acc = float(sum(f1_list) / len(f1_list)) if f1_list else 0.0
    avg_cost = float(total_cost / len(outputs)) if outputs else 0.0
    return {
        "accuracy": acc,
        "avg_cost": avg_cost,
        "failures": failures,
        "num_samples": len(outputs),
        "prompt_config": prompt_config,
    }



# ---------------------------------------------------------------------------
# Prompt optimization (mirrors MMLU/IFEval pattern)
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
    for mutation in DROP_PROMPT_MUTATIONS:
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


def _diagnose_failures(agent, failures):
    """Ask the model to produce a one-line diagnosis per failure."""
    if not failures:
        return failures
    preview = []
    for f in failures[:4]:
        preview.append({
            "question": f["question"][:600],
            "pred": f["pred"],
            "gold": f["gold"],
        })
    try:
        response = agent.action_call_json_format_llm(
            model=getattr(agent, "optimization_model",
                          getattr(agent, "default_infer_model", "gpt-4.1-mini")),
            messages=[{
                "role": "user",
                "content": (
                    "For each failed DROP reading comprehension example below, write a brief "
                    "one-sentence diagnosis explaining WHY the model likely got the wrong answer.\n"
                    f"Failures:\n{json.dumps(preview, ensure_ascii=True, indent=2)}"
                ),
            }],
            temperature=0.3,
            num_of_response=1,
            role="error analyst",
            return_dict_keys=["diagnoses"],
            requirements=(
                "Return key 'diagnoses' as a list of strings, one per failure. "
                "Each diagnosis should be a single sentence identifying the likely reasoning error."
            ),
        )
        payload = response[0] if response else {}
        diagnoses = payload.get("diagnoses", []) if isinstance(payload, dict) else []
        if isinstance(diagnoses, str):
            diagnoses = [diagnoses]
    except Exception:
        diagnoses = []

    enriched = []
    for i, f in enumerate(failures):
        ef = dict(f)
        if i < len(diagnoses) and isinstance(diagnoses[i], str):
            ef["diagnosis"] = diagnoses[i]
        else:
            ef["diagnosis"] = f"Predicted '{f['pred']}' but correct was '{f['gold']}'."
        enriched.append(ef)
    return enriched


def _generate_reflective_candidates(agent, base_prompt, failures, max_new=2):
    if not failures or max_new <= 0:
        return []

    diagnosed = _diagnose_failures(agent, failures[:3])
    failure_preview = []
    for f in diagnosed:
        failure_preview.append({
            "question": f["question"][:800],
            "pred": f["pred"],
            "gold": f["gold"],
            "diagnosis": f.get("diagnosis", ""),
        })

    reflection_messages = [
        {
            "role": "user",
            "content": (
                "You are optimizing a DROP reading comprehension prompt for a small model. "
                "Given current prompt config and failed examples with diagnoses, propose diverse prompt improvements.\n"
                "Each diagnosis explains WHY the model failed — use these to target your improvements.\n"
                "Return full revised prompt configs, not suffixes or patches. "
                "If you change requirements, provide the complete replacement requirements string.\n"
                f"Current prompt config:\n{json.dumps(base_prompt, ensure_ascii=True, indent=2)}\n"
                f"Failed examples with diagnoses:\n{json.dumps(failure_preview, ensure_ascii=True, indent=2)}"
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


def _generate_merged_candidate(agent, cand_a, cand_b):
    """Merge strengths of two complementary Pareto candidates."""
    try:
        response = agent.action_call_json_format_llm(
            model=getattr(agent, "optimization_model",
                          getattr(agent, "default_infer_model", "gpt-4.1-mini")),
            messages=[{
                "role": "user",
                "content": (
                    "Two DROP prompt configs each solve different questions well. "
                    "Merge their strengths into a single prompt config that combines the best of both.\n"
                    f"Config A (accuracy={cand_a['accuracy']:.3f}):\n"
                    f"{json.dumps(cand_a['prompt_config'], ensure_ascii=True, indent=2)}\n\n"
                    f"Config B (accuracy={cand_b['accuracy']:.3f}):\n"
                    f"{json.dumps(cand_b['prompt_config'], ensure_ascii=True, indent=2)}"
                ),
            }],
            temperature=0.5,
            num_of_response=1,
            role="prompt merger",
            return_dict_keys=["role", "requirements", "temperature"],
            requirements=(
                "Return a single merged prompt config with keys: role, requirements, temperature. "
                "Combine the strongest elements from both configs."
            ),
        )
        payload = response[0] if response else {}
        if isinstance(payload, dict) and ("role" in payload or "requirements" in payload):
            return normalize_prompt_config(payload)
    except Exception:
        pass
    return None


def _select_parent_from_pareto(pareto_front, rng):
    if len(pareto_front) == 1:
        return pareto_front[0]
    weights = [len(c.get("failures", [])) + 1 for c in pareto_front]
    total = sum(weights)
    weights = [w / total for w in weights]
    return rng.choices(pareto_front, weights=weights, k=1)[0]


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
    examples = _load_drop_examples()
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
        iter_rng = random.Random(seed + iter_idx)
        parent = _select_parent_from_pareto(pareto_front, iter_rng)
        handcrafted = _generate_handcrafted_candidates(
            parent["prompt_config"],
            candidates_per_iter=max(2, int(candidates_per_iter)),
        )
        reflective = _generate_reflective_candidates(
            agent,
            parent["prompt_config"],
            parent["failures"],
            max_new=max(0, int(candidates_per_iter) - len(handcrafted) + 1),
        )

        merged_candidates = []
        if len(pareto_front) >= 2:
            sorted_front = sorted(pareto_front, key=lambda x: x["accuracy"])
            merged = _generate_merged_candidate(agent, sorted_front[0], sorted_front[-1])
            if merged is not None:
                merged_candidates.append(merged)
                print(f"[inner-loop] Iter {iter_idx+1}/{iterations}: "
                      f"generated 1 merged candidate from Pareto crossover", flush=True)

        raw_candidates = handcrafted + reflective + merged_candidates
        dedup = []
        seen = set()
        for cand in raw_candidates:
            key = json.dumps(cand, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(cand)

        print(f"[inner-loop] Iter {iter_idx+1}/{iterations}: "
              f"{len(handcrafted)} handcrafted + {len(reflective)} reflective + "
              f"{len(merged_candidates)} merged → "
              f"{len(dedup)} unique candidates", flush=True)

        eval_results = []
        for cand_idx, cand in enumerate(dedup):
            cand_eval = _evaluate_prompt_config(agent, cand, train_examples)
            eval_results.append(cand_eval)
            print(f"[inner-loop] Iter {iter_idx+1}/{iterations}: "
                  f"candidate {cand_idx+1}/{len(dedup)} → "
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

def real_evaluate(solver):
    examples = _load_drop_examples()[128:928]
    questions = [ex["inputs"] for ex in examples]
    answers = [ex["targets"] for ex in examples]

    max_workers = min(len(examples), 48)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(wrap_solver(solver), questions), total=len(questions)))

    acc_list = []
    info_list = []
    for q_idx, res in enumerate(results):
        try:
            extracted_answer = str(res.get("answer", "NO ANSWER IN DICTIONARY"))
            correct_answers = answers[q_idx]
            em_score, f1_score = drop_metric(extracted_answer, correct_answers)
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(em_score)
        info_list.append(
            f"Sample {q_idx}:\n{questions[q_idx]}\nModel Output: {res}\n"
            f"Model Answer: {extracted_answer}\nCorrect Answers: {correct_answers}\n"
            f"EM: {em_score}  F1: {f1_score}\n"
        )

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"results/drop_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc


class DROP_Task:
    def evaluate(self, solver, examples_override=None, max_workers=48):
        if examples_override is not None:
            examples = list(examples_override)
        else:
            examples = _load_drop_examples()[:128]
            random.seed(time.time())
            random.shuffle(examples)
            examples = examples[:40]

        questions = [ex["inputs"] for ex in examples]
        answers = [ex["targets"] for ex in examples]

        max_workers = min(len(examples), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(solver, questions), total=len(questions)))

        acc_list = []
        info_list = []
        for q_idx, res in enumerate(results):
            try:
                extracted_answer = str(res.get("answer", "NO ANSWER IN DICTIONARY"))
                correct_answers = answers[q_idx]
                em_score, f1_score = drop_metric(extracted_answer, correct_answers)
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(em_score)
            info_list.append(
                f"Valid Sample {q_idx}:\n{questions[q_idx]}\nModel Output: {res}\n"
                f"Model Answer: {extracted_answer}\nCorrect Answers: {correct_answers}\n"
                f"EM: {em_score}  F1: {f1_score}\n"
            )

        valid_acc = sum(acc_list) / len(acc_list)
        print("EM Acc:", valid_acc)

        global force_first_full_eval
        global last_test_acc
        should_run_test = force_first_full_eval or valid_acc >= threshold

        if should_run_test:
            test_acc = real_evaluate(solver)
            last_test_acc = test_acc
            gate_note = ""
            if force_first_full_eval and valid_acc < threshold:
                gate_note = (
                    f"Forced first full evaluation despite valid EM accuracy below {threshold}.\n"
                )
            force_first_full_eval = False
            feedback = (
                f"Valid EM Accuracy: {valid_acc}\n"
                + gate_note
                + f"Test EM Accuracy: {test_acc}\n"
                + "Evaluation Info:\n"
                + "\n".join(info_list)
            )
        else:
            test_acc = last_test_acc
            feedback = (
                f"Valid EM Accuracy: {valid_acc}\nValid EM Accuracy less than {threshold}, no testing needed.\n"
                f"Cached Test EM Accuracy: {test_acc}\n"
                + "Evaluation Info:\n"
                + "\n".join(info_list)
            )
        return feedback, test_acc


def bootstrap_confidence_interval(data, confidence_level=0.95):
    n = len(data)
    if n == 0:
        return "95% Confidence Interval: (0.0%, 0.0%), Median: 0.0%"

    mean = sum(data) / n
    z = 1.96
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
