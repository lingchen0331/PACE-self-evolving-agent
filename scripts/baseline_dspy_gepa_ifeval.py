#!/usr/bin/env python3
"""
DSPy GEPA baseline for IFEval — comparable to BAE experiments.

Usage:
    source .venv/bin/activate
    python scripts/baseline_dspy_gepa_ifeval.py [--model ministral-14b-local] [--auto medium]

Requires two vLLM servers on ports 8000/8001 (see scripts/serve_all.sh).
"""
import argparse
import itertools
import json
import random
import re
import threading
import time
from pathlib import Path

import dspy


# ── Round-robin LM wrapper ───────────────────────────────────────────
class RoundRobinLM(dspy.LM):
    """Dispatches calls across multiple OpenAI-compatible endpoints.

    DSPy threads each example through a single LM instance; by rotating
    api_base per call, we fan out across vLLM replicas without changing
    any downstream code.
    """
    def __init__(self, model: str, base_urls, **kwargs):
        super().__init__(model=model, api_base=base_urls[0], **kwargs)
        self._base_urls = list(base_urls)
        self._cycle = itertools.cycle(self._base_urls)
        self._lock = threading.Lock()

    def _next_base(self):
        with self._lock:
            return next(self._cycle)

    def forward(self, prompt=None, messages=None, **kwargs):
        kwargs["api_base"] = self._next_base()
        return super().forward(prompt=prompt, messages=messages, **kwargs)


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DSPy GEPA baseline on IFEval")
    p.add_argument("--model", default="ministral-14b-local")
    p.add_argument("--base-urls", nargs="+",
                   default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"])
    p.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                   help="GEPA budget preset")
    p.add_argument("--train-size", type=int, default=200,
                   help="Number of training examples for GEPA")
    p.add_argument("--val-size", type=int, default=100,
                   help="Number of validation examples for GEPA")
    p.add_argument("--test-size", type=int, default=141,
                   help="Number of test examples (IFEval test split = 141)")
    p.add_argument("--num-threads", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Max output tokens (Qwen3 thinking traces need ample room)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/dspy_gepa_ifeval")
    return p.parse_args()


# ── Instruction Verification (mirrors src/task_ifeval.py) ────────────
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


INSTRUCTION_CHECKERS = {
    "punctuation:no_comma": lambda r, **kw: "," not in r,
    "startend:end_checker": lambda r, **kw: r.rstrip().endswith(kw.get("end_phrase", "")),
    "startend:quotation": lambda r, **kw: r.strip().startswith('"') and r.strip().endswith('"'),
    "change_case:english_lowercase": lambda r, **kw: r == r.lower(),
    "change_case:english_capital": lambda r, **kw: r == r.upper(),
    "change_case:capital_word_frequency": lambda r, **kw: _check_relation(
        sum(1 for w in r.split() if w.isupper() and w.isalpha()),
        kw.get("capital_relation", "at least"), int(kw.get("capital_frequency", 1))),
    "length_constraints:number_words": lambda r, **kw: _check_relation(
        len(r.split()), kw.get("relation", "at least"), int(kw.get("num_words", 1))),
    "length_constraints:number_sentences": lambda r, **kw: _check_relation(
        len([s for s in re.split(r'[.!?]+', r.strip()) if s.strip()]),
        kw.get("relation", "at least"), int(kw.get("num_sentences", 1))),
    "length_constraints:number_paragraphs": lambda r, **kw:
        len([p for p in r.split("\n\n") if p.strip()]) >= int(kw.get("num_paragraphs", 1)),
    "detectable_format:number_bullet_lists": lambda r, **kw:
        len(re.findall(r'^\s*[\*\-•]\s', r, re.MULTILINE)) >= int(kw.get("num_bullets", 1)),
    "detectable_format:number_highlighted_sections": lambda r, **kw:
        len(re.findall(r'\*[^*\n]+\*', r)) >= int(kw.get("num_highlights", 1)),
    "detectable_format:title": lambda r, **kw: bool(re.search(r'<<[^>]+>>', r)),
    "detectable_content:postscript": lambda r, **kw: kw.get("postscript_marker", "P.S.") in r,
    "keywords:existence": lambda r, **kw: all(
        k.lower() in r.lower() for k in kw.get("keywords", [])),
    "keywords:forbidden_words": lambda r, **kw: all(
        fw.lower() not in r.lower() for fw in kw.get("forbidden_words", [])),
    "keywords:frequency": lambda r, **kw: _check_relation(
        r.lower().count(kw.get("keyword", "").lower()),
        kw.get("relation", "at least"), int(kw.get("frequency", 1))),
    "language:response_language": lambda r, **kw: len(r.strip()) > 0,
}


def verify_all_instructions(response_text, instruction_id_list, kwargs_list):
    results = []
    for iid, kw in zip(instruction_id_list, kwargs_list):
        checker = INSTRUCTION_CHECKERS.get(iid)
        if checker is None:
            results.append(True)
            continue
        try:
            results.append(checker(response_text, **kw))
        except Exception:
            results.append(False)
    strict = 1 if all(results) else 0
    loose = sum(results) / len(results) if results else 0.0
    return strict, loose, results


# ── Data ─────────────────────────────────────────────────────────────
def load_ifeval_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_ifeval_splits(train_n, val_n, test_n, seed=0):
    train_raw = load_ifeval_jsonl("datasets/ifeval/ifeval_input_data_train.jsonl")
    test_raw = load_ifeval_jsonl("datasets/ifeval/ifeval_input_data_test.jsonl")

    rng = random.Random(seed)
    rng.shuffle(train_raw)

    # Use train split for GEPA train+val, test split for final eval
    total_train_needed = train_n + val_n
    if total_train_needed > len(train_raw):
        print(f"Warning: requested {total_train_needed} train+val but only {len(train_raw)} available, adjusting.")
        train_n = min(train_n, len(train_raw) * 2 // 3)
        val_n = min(val_n, len(train_raw) - train_n)

    test_n = min(test_n, len(test_raw))

    def to_example(row):
        return dspy.Example(
            prompt=row["prompt"],
            instruction_id_list=row["instruction_id_list"],
            kwargs=row["kwargs"],
        ).with_inputs("prompt")

    train = [to_example(r) for r in train_raw[:train_n]]
    val = [to_example(r) for r in train_raw[train_n:train_n + val_n]]
    test = [to_example(r) for r in test_raw[:test_n]]
    return train, val, test


# ── DSPy Module ──────────────────────────────────────────────────────
class IFEvalResponse(dspy.Signature):
    """Follow the user's instructions precisely, satisfying ALL formatting, keyword, length, and style constraints."""
    prompt: str = dspy.InputField(desc="The user's instruction with constraints to follow")
    reasoning: str = dspy.OutputField(desc="Step-by-step analysis of each constraint that must be satisfied")
    response: str = dspy.OutputField(desc="A response that satisfies every constraint in the prompt")


class IFEvalSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(IFEvalResponse)

    def forward(self, prompt: str):
        return self.predict(prompt=prompt)


# ── Metric ───────────────────────────────────────────────────────────
def ifeval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    response_text = str(getattr(pred, "response", "") or "")
    instruction_ids = gold.instruction_id_list
    kwargs_list = gold.kwargs

    strict, loose, per_instr = verify_all_instructions(response_text, instruction_ids, kwargs_list)

    if pred_name is not None:
        failed_ids = [iid for iid, passed in zip(instruction_ids, per_instr) if not passed]
        if strict:
            feedback = "All instructions satisfied."
        else:
            feedback = (
                f"Failed {len(failed_ids)}/{len(instruction_ids)} instructions: {failed_ids}. "
                f"Loose accuracy: {loose:.2f}. "
                f"The response must satisfy ALL constraints simultaneously."
            )
        return dspy.Prediction(score=float(strict), feedback=feedback)

    return float(strict)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure LM — round-robin across vLLM replicas
    lm = RoundRobinLM(
        model=f"openai/{args.model}",
        base_urls=args.base_urls,
        api_key="EMPTY",
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=lm)

    reflection_lm = RoundRobinLM(
        model=f"openai/{args.model}",
        base_urls=args.base_urls,
        api_key="EMPTY",
        temperature=1.0,
        max_tokens=args.max_tokens,
    )

    # Load data
    print(f"Loading IFEval splits: train={args.train_size}, val={args.val_size}, test={args.test_size}")
    trainset, valset, testset = load_ifeval_splits(
        args.train_size, args.val_size, args.test_size, seed=args.seed
    )
    print(f"  Actual sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    student = IFEvalSolver()

    # ── Baseline evaluation ──────────────────────────────────────────
    print("\n========== Baseline (no optimization) ==========")
    baseline_eval = dspy.Evaluate(
        devset=testset, metric=ifeval_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
    )
    baseline_result = baseline_eval(student)
    baseline_score = float(baseline_result)
    print(f"Baseline strict accuracy: {baseline_score:.4f}")

    # ── GEPA optimization ────────────────────────────────────────────
    print(f"\n========== GEPA optimization (auto={args.auto}) ==========")
    t0 = time.time()

    gepa = dspy.GEPA(
        metric=ifeval_metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        num_threads=args.num_threads,
        track_stats=True,
        log_dir=str(out_dir / "gepa_logs"),
        seed=args.seed,
    )

    optimized = gepa.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )
    opt_time = time.time() - t0
    print(f"Optimization took {opt_time:.1f}s")

    # ── Post-optimization evaluation ─────────────────────────────────
    print("\n========== Optimized evaluation ==========")
    opt_eval = dspy.Evaluate(
        devset=testset, metric=ifeval_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
    )
    opt_result = opt_eval(optimized)
    opt_score = float(opt_result)
    print(f"Optimized strict accuracy: {opt_score:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "method": "DSPy-GEPA",
        "model": args.model,
        "task": "ifeval",
        "auto": args.auto,
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "baseline_strict_accuracy": baseline_score,
        "optimized_strict_accuracy": opt_score,
        "optimization_time_s": round(opt_time, 1),
        "seed": args.seed,
    }

    if hasattr(optimized, "detailed_results"):
        dr = optimized.detailed_results
        results["total_metric_calls"] = dr.total_metric_calls
        results["num_full_val_evals"] = dr.num_full_val_evals
        results["best_val_score"] = max(dr.val_aggregate_scores) if dr.val_aggregate_scores else None
        results["num_candidates_explored"] = len(dr.candidates)

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    optimized.save(str(out_dir / "optimized_program.json"))
    print(f"Optimized program saved to {out_dir / 'optimized_program.json'}")

    print("\n========== Summary ==========")
    print(f"  Baseline:  {baseline_score:.4f}")
    print(f"  Optimized: {opt_score:.4f}")
    print(f"  Delta:     {opt_score - baseline_score:+.4f}")


if __name__ == "__main__":
    main()
