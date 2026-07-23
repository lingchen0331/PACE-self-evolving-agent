import copy
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas
from tqdm import tqdm

from wrap import wrap_solver

threshold = 0.80
last_test_acc = 0.0

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question.
Question's Subject: {Subject}
Question: {Question}

Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}
""".strip()

DEFAULT_PROMPT_CONFIG = {
    "role": "knowledge and reasoning expert",
    "requirements": (
        "1. Analyze the question and options carefully.\n"
        "2. Return JSON with keys reasoning and answer.\n"
        "3. The answer MUST be exactly one of: A, B, C, D."
    ).strip(),
    "temperature": 0.2,
}

MMLU_PROMPT_MUTATIONS = [
    {
        "requirements_suffix": (
            "4. First eliminate obviously wrong options, then choose the best remaining answer."
        )
    },
    {
        "requirements_suffix": (
            "4. Prefer direct factual recall before abstract speculation."
        )
    },
    {
        "requirements_suffix": (
            "4. If uncertain, compare all options and pick the one with the strongest evidence."
        )
    },
    {
        "temperature": 0.1,
    },
    {
        "temperature": 0.0,
    },
    {
        "role": "MMLU specialist focused on academic multiple-choice accuracy",
    },
]


def solver(agent, task: str):
    prompt_cfg = normalize_prompt_config(getattr(agent, "prompt_config", DEFAULT_PROMPT_CONFIG))
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

    return _coerce_solver_output(task, response[0] if response else {})


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def _load_mmlu_examples():
    data_filename = "datasets/mmlu.csv"
    df = pandas.read_csv(data_filename, engine="python", on_bad_lines="skip")
    random.seed(0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)
    return examples


def _parse_choices_from_task(task_text):
    choices = {}
    for match in re.finditer(r"^\(([ABCD])\)\s*(.+)$", str(task_text), flags=re.MULTILINE):
        choices[match.group(1)] = match.group(2).strip()
    return choices


def _normalize_choice_text(text):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _extract_answer(raw_answer, task_text=None):
    extracted_answer = str(raw_answer or "").strip()
    if not extracted_answer:
        return extracted_answer

    direct_patterns = [
        r"^\s*([ABCD])\s*$",
        r"^\s*\(?([ABCD])\)?[\s\.:,-]*$",
        r"\banswer\s*(?:is|:)?\s*\(?([ABCD])\)?\b",
        r"\boption\s*([ABCD])\b",
        r"\bchoice\s*([ABCD])\b",
        r"\(([ABCD])\)",
    ]
    for pattern in direct_patterns:
        match = re.search(pattern, extracted_answer, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if extracted_answer.startswith(("A", "B", "C", "D")):
        return extracted_answer[0]

    choices = _parse_choices_from_task(task_text) if task_text is not None else {}
    normalized_answer = _normalize_choice_text(extracted_answer)
    for letter, choice_text in choices.items():
        normalized_choice = _normalize_choice_text(choice_text)
        if not normalized_choice:
            continue
        if normalized_answer == normalized_choice:
            return letter
        if normalized_answer.endswith(normalized_choice):
            return letter
        if len(normalized_choice) >= 4 and normalized_choice in normalized_answer:
            return letter

    return extracted_answer


def _coerce_solver_output(task_text, output):
    if isinstance(output, dict):
        return_dict = dict(output)
    else:
        return_dict = {"answer": str(output or "")}

    reasoning = str(return_dict.get("reasoning", "")).strip()
    candidates = [
        return_dict.get("answer", ""),
        return_dict.get("response", ""),
        reasoning,
    ]

    resolved_answer = ""
    for candidate in candidates:
        resolved_answer = _extract_answer(candidate, task_text=task_text)
        if resolved_answer in {"A", "B", "C", "D"}:
            break

    return_dict["reasoning"] = reasoning
    return_dict["answer"] = resolved_answer or str(return_dict.get("answer", "")).strip()
    return return_dict


def _single_example_eval(agent, question, prompt_config):
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
        output = _coerce_solver_output(question, response[0] if response else {})
        reasoning = str(output.get("reasoning", ""))
        answer = _extract_answer(output.get("answer", ""), task_text=question)
        return {
            "ok": True,
            "reasoning": reasoning,
            "answer": answer,
            "cost": len(reasoning) + len(answer),
            "raw": output,
        }
    except Exception as e:
        return {
            "ok": False,
            "reasoning": "",
            "answer": "",
            "cost": 10000,
            "raw": repr(e),
        }


def _evaluate_prompt_config(agent, prompt_config, examples, max_workers=16):
    questions = [format_multichoice_question(example) for example in examples]
    answers = [example["Answer"] for example in examples]

    prompt_config = normalize_prompt_config(prompt_config)
    worker = max(1, min(max_workers, len(questions)))

    with ThreadPoolExecutor(max_workers=worker) as executor:
        outputs = list(
            executor.map(
                lambda q: _single_example_eval(agent, q, prompt_config),
                questions,
            )
        )

    acc_list = []
    total_cost = 0
    failures = []
    for idx, out in enumerate(outputs):
        correct_answer = str(answers[idx])
        is_correct = out["ok"] and out["answer"] == correct_answer
        acc_list.append(1 if is_correct else 0)
        total_cost += out["cost"]
        if not is_correct and len(failures) < 6:
            failures.append(
                {
                    "question": questions[idx],
                    "pred": out.get("answer", ""),
                    "gold": correct_answer,
                    "raw": out.get("raw", ""),
                }
            )

    acc = float(sum(acc_list) / len(acc_list)) if acc_list else 0.0
    avg_cost = float(total_cost / len(outputs)) if outputs else 0.0
    return {
        "accuracy": acc,
        "avg_cost": avg_cost,
        "failures": failures,
        "num_samples": len(outputs),
        "prompt_config": prompt_config,
    }


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
    for mutation in MMLU_PROMPT_MUTATIONS:
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
    """Ask the SLM to produce a one-line diagnosis per failure (GEPA-style trace reflection)."""
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
            model=getattr(agent, "optimization_model", getattr(agent, "default_infer_model", "gpt-4.1-mini")),
            messages=[{
                "role": "user",
                "content": (
                    "For each failed MMLU example below, write a brief one-sentence diagnosis "
                    "explaining WHY the model likely chose the wrong answer.\n"
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

    # GEPA-inspired: enrich failures with per-example diagnosis
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
                "You are optimizing an MMLU prompt for a small model. "
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
            model=getattr(agent, "optimization_model", getattr(agent, "default_infer_model", "gpt-4.1-mini")),
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
    """GEPA-inspired merge: combine strengths of two complementary Pareto candidates."""
    try:
        response = agent.action_call_json_format_llm(
            model=getattr(agent, "optimization_model", getattr(agent, "default_infer_model", "gpt-4.1-mini")),
            messages=[{
                "role": "user",
                "content": (
                    "Two MMLU prompt configs each solve different questions well. "
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
    """GEPA-inspired: sample parent proportional to unique-failure coverage (diversity)."""
    if len(pareto_front) == 1:
        return pareto_front[0]
    # Weight by number of unique failures — candidates with more failures
    # cover different error modes and benefit most from mutation.
    # Add 1 so candidates with 0 failures still have a small chance.
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
    examples = _load_mmlu_examples()
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
        # GEPA-inspired: sample parent from Pareto front weighted by coverage diversity
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

        # GEPA-inspired merge: if Pareto front has 2+ candidates, try crossover
        merged_candidates = []
        if len(pareto_front) >= 2:
            # Pick two most diverse candidates (highest and lowest accuracy on front)
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
        history.append(
            {
                "stage": "iteration",
                "accuracy": iter_best["accuracy"],
                "avg_cost": iter_best["avg_cost"],
                "prompt_config": iter_best["prompt_config"],
            }
        )

    print(f"[inner-loop] Validation phase: evaluating baseline + {len(pareto_front)} pareto candidates "
          f"on {len(valid_examples)} valid samples", flush=True)
    initial_valid_eval = _evaluate_prompt_config(agent, current_prompt_config, valid_examples)
    print(f"[inner-loop] Baseline valid accuracy: {initial_valid_eval['accuracy']:.4f}", flush=True)

    valid_scored = []
    for vi, cand in enumerate(pareto_front):
        valid_eval = _evaluate_prompt_config(agent, cand["prompt_config"], valid_examples)
        print(f"[inner-loop] Valid candidate {vi+1}/{len(pareto_front)}: "
              f"acc={valid_eval['accuracy']:.4f}", flush=True)
        valid_scored.append(
            {
                "train_accuracy": cand["accuracy"],
                "train_avg_cost": cand["avg_cost"],
                "valid_accuracy": valid_eval["accuracy"],
                "valid_avg_cost": valid_eval["avg_cost"],
                "prompt_config": cand["prompt_config"],
            }
        )

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


def real_evaluate(solver):
    examples = _load_mmlu_examples()[128:928]
    questions = [format_multichoice_question(example) for example in examples]
    answers = [example["Answer"] for example in examples]

    max_workers = min(len(examples), 48)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(wrap_solver(solver), questions), total=len(questions)))

    acc_list = []
    info_list = []
    for q_idx, res in enumerate(results):
        try:
            extracted_answer = _extract_answer(res["answer"])
            correct_answer = str(answers[q_idx])
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(extracted_answer == correct_answer)
        info_list.append(
            f"Sample {q_idx}:\n{questions[q_idx]}\nModel Output: {res}\n"
            f"Model Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n"
        )

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"results/mmlu_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc


class MMLU_Task:
    def evaluate(self, solver, examples_override=None, max_workers=48):
        if examples_override is not None:
            examples = list(examples_override)
        else:
            examples = _load_mmlu_examples()[:128]
            random.seed(time.time())
            random.shuffle(examples)
            examples = examples[:20]

        questions = [format_multichoice_question(example) for example in examples]
        answers = [example["Answer"] for example in examples]

        max_workers = min(len(examples), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(solver, questions), total=len(questions)))

        acc_list = []
        info_list = []
        for q_idx, res in enumerate(results):
            try:
                extracted_answer = _extract_answer(res["answer"])
                correct_answer = str(answers[q_idx])
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(extracted_answer == correct_answer)
            info_list.append(
                f"Valid Sample {q_idx}:\n{questions[q_idx]}\nModel Output: {res}\n"
                f"Model Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n"
            )

        valid_acc = sum(acc_list) / len(acc_list)
        print("Acc:", valid_acc)
        if valid_acc >= threshold:
            test_acc = real_evaluate(solver)
            feedback = (
                f"Valid Accuracy: {valid_acc}\nTest Accuracy {test_acc}\n"
                + "Evaluation Info:\n"
                + "\n".join(info_list)
            )
        else:
            test_acc = 0
            feedback = (
                f"Valid Accuracy: {valid_acc}\nValid Accuracy less than {threshold}, no testing needed.\n"
                + "Evaluation Info:\n"
                + "\n".join(info_list)
            )
        return feedback, test_acc


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.

    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    data = np.array(data)

    bootstrap_means = []
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    bootstrap_means = np.array(bootstrap_means)

    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)
    median = np.median(bootstrap_means)

    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    print(
        f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), "
        f"Median: {median_percent:.1f}%"
    )
    return (
        f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), "
        f"Median: {median_percent:.1f}%"
    )
