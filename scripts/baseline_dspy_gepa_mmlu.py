#!/usr/bin/env python3
"""
DSPy GEPA baseline for MMLU — comparable to BAE experiments.

Usage:
    source .venv/bin/activate
    python scripts/baseline_dspy_gepa_mmlu.py [--model ministral-14b-local] [--auto medium]

Requires two vLLM servers on ports 8000/8001 (see scripts/serve_all.sh).
"""
import argparse
import itertools
import json
import random
import threading
import time
from pathlib import Path

import dspy
import pandas as pd


# ── Round-robin LM wrapper ───────────────────────────────────────────
class RoundRobinLM(dspy.LM):
    """Dispatches calls across multiple OpenAI-compatible endpoints.

    DSPy threads each example through a single LM instance; by rotating
    api_base per call, we fan out across vLLM replicas without changing
    any downstream code.
    """
    def __init__(self, model: str, base_urls, **kwargs):
        # Initialise with the first URL so parent class sets everything up.
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
    p = argparse.ArgumentParser(description="DSPy GEPA baseline on MMLU")
    p.add_argument("--model", default="ministral-14b-local")
    p.add_argument("--base-urls", nargs="+",
                   default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"])
    p.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                   help="GEPA budget preset")
    p.add_argument("--train-size", type=int, default=200,
                   help="Number of training examples for GEPA")
    p.add_argument("--val-size", type=int, default=100,
                   help="Number of validation examples for GEPA")
    p.add_argument("--test-size", type=int, default=800,
                   help="Number of test examples (paper uses 800)")
    p.add_argument("--num-threads", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Max output tokens (Qwen3 thinking traces need ample room)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/dspy_gepa_mmlu")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────
QUERY_TEMPLATE = (
    "Answer the following multiple choice question.\n"
    "Subject: {Subject}\n"
    "Question: {Question}\n\n"
    "Choices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}"
)


def load_mmlu_splits(train_n, val_n, test_n, seed=0):
    df = pd.read_csv("datasets/mmlu.csv", engine="python", on_bad_lines="skip")
    rows = [r.to_dict() for _, r in df.iterrows()]
    rng = random.Random(seed)
    rng.shuffle(rows)

    total_needed = train_n + val_n + test_n
    if total_needed > len(rows):
        raise ValueError(f"Need {total_needed} examples but only {len(rows)} available")

    def to_example(row):
        question_text = QUERY_TEMPLATE.format(**row)
        return dspy.Example(
            question=question_text,
            answer=str(row["Answer"]).strip().upper(),
        ).with_inputs("question")

    train = [to_example(r) for r in rows[:train_n]]
    val = [to_example(r) for r in rows[train_n:train_n + val_n]]
    test = [to_example(r) for r in rows[train_n + val_n:train_n + val_n + test_n]]
    return train, val, test


# ── DSPy Module ──────────────────────────────────────────────────────
class MMLUAnswer(dspy.Signature):
    """Answer a multiple-choice question. Return exactly one letter: A, B, C, or D."""
    question: str = dspy.InputField(desc="The multiple-choice question with options")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
    answer: str = dspy.OutputField(desc="Exactly one of: A, B, C, D")


class MMLUSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(MMLUAnswer)

    def forward(self, question: str):
        return self.predict(question=question)


# ── Metric ───────────────────────────────────────────────────────────
def normalize_answer(ans: str) -> str:
    """Extract a single letter A-D from model output."""
    ans = str(ans).strip().upper()
    for ch in ans:
        if ch in "ABCD":
            return ch
    return ""


def mmlu_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    predicted = normalize_answer(pred.answer)
    expected = normalize_answer(gold.answer)
    correct = predicted == expected

    if pred_name is not None:
        # Provide richer feedback for GEPA reflection
        if correct:
            feedback = f"Correct! The answer is {expected}."
        else:
            feedback = (
                f"Incorrect. Predicted '{predicted}' but expected '{expected}'. "
                f"The model's reasoning may have a factual error or logical gap."
            )
        return dspy.Prediction(score=1.0 if correct else 0.0, feedback=feedback)

    return 1.0 if correct else 0.0


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

    # Reflection LM — same round-robin pool, higher temperature for diversity
    reflection_lm = RoundRobinLM(
        model=f"openai/{args.model}",
        base_urls=args.base_urls,
        api_key="EMPTY",
        temperature=1.0,
        max_tokens=args.max_tokens,
    )

    # Load data
    print(f"Loading MMLU splits: train={args.train_size}, val={args.val_size}, test={args.test_size}")
    trainset, valset, testset = load_mmlu_splits(
        args.train_size, args.val_size, args.test_size, seed=args.seed
    )

    # Build student
    student = MMLUSolver()

    # ── Baseline (pre-optimization) evaluation ───────────────────────
    print("\n========== Baseline (no optimization) ==========")
    baseline_eval = dspy.Evaluate(
        devset=testset, metric=mmlu_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
    )
    baseline_result = baseline_eval(student)
    baseline_score = float(baseline_result)
    print(f"Baseline accuracy: {baseline_score:.4f}")

    # ── GEPA optimization ────────────────────────────────────────────
    print(f"\n========== GEPA optimization (auto={args.auto}) ==========")
    t0 = time.time()

    gepa = dspy.GEPA(
        metric=mmlu_metric,
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
        devset=testset, metric=mmlu_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
    )
    opt_result = opt_eval(optimized)
    opt_score = float(opt_result)
    print(f"Optimized accuracy: {opt_score:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "method": "DSPy-GEPA",
        "model": args.model,
        "task": "mmlu",
        "auto": args.auto,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "baseline_accuracy": baseline_score,
        "optimized_accuracy": opt_score,
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

    # Save optimized program
    optimized.save(str(out_dir / "optimized_program.json"))
    print(f"Optimized program saved to {out_dir / 'optimized_program.json'}")

    # Print summary
    print("\n========== Summary ==========")
    print(f"  Baseline:  {baseline_score:.4f}")
    print(f"  Optimized: {opt_score:.4f}")
    print(f"  Delta:     {opt_score - baseline_score:+.4f}")


if __name__ == "__main__":
    main()
