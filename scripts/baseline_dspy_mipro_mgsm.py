#!/usr/bin/env python3
"""
DSPy MIPROv2 baseline for MGSM — comparable to BAE experiments.

Usage:
    source .venv/bin/activate
    python scripts/baseline_dspy_mipro_mgsm.py [--model ministral-14b-local] [--auto medium]

Requires vLLM servers on ports 8000+ (see scripts/serve_all.sh).
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
    """Dispatches calls across multiple OpenAI-compatible endpoints."""
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        del state["_cycle"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(self._base_urls)

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("_lock", "_cycle"):
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        result._lock = threading.Lock()
        result._cycle = itertools.cycle(result._base_urls)
        return result


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DSPy MIPROv2 baseline on MGSM")
    p.add_argument("--model", default="ministral-14b-local")
    p.add_argument("--base-urls", nargs="+",
                   default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
                            "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"])
    p.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                   help="MIPROv2 budget preset")
    p.add_argument("--train-size", type=int, default=200,
                   help="Number of training examples for MIPROv2")
    p.add_argument("--val-size", type=int, default=100,
                   help="Number of validation examples for MIPROv2")
    p.add_argument("--test-size", type=int, default=800,
                   help="Number of test examples")
    p.add_argument("--num-threads", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Max output tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/dspy_mipro_mgsm")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────
ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

LANG_TO_INSTRUCTIONS = {
    "en": "Solve this math problem.\n\n{input}",
    "bn": "এই গণিতের সমস্যাটি সমাধান করুন।\n\n{input}",
    "de": "Löse dieses Mathematikproblem.\n\n{input}",
    "es": "Resuelve este problema matemático.\n\n{input}",
    "fr": "Résolvez ce problème de mathématiques.\n\n{input}",
    "ja": "この数学の問題を解いてください。\n\n{input}",
    "ru": "Решите эту математическую задачу.\n\n{input}",
    "sw": "Suluhisha tatizo hili la hesabu.\n\n{input}",
    "te": "ఈ గణిత సమస్యను పరిష్కరించండి.\n\n{input}",
    "th": "แก้ปัญหาคณิตศาสตร์นี้\n\n{input}",
    "zh": "解决这个数学问题。\n\n{input}",
}


def load_all_mgsm_examples():
    examples = []
    for lang in ALL_LANGUAGES:
        fpath = f"datasets/mgsm/mgsm_{lang}.tsv"
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                inputs, targets = parts
                question = LANG_TO_INSTRUCTIONS[lang].format(input=inputs)
                examples.append({"question": question, "answer": targets, "lang": lang})
    return examples


def load_mgsm_splits(train_n, val_n, test_n, seed=0):
    rows = load_all_mgsm_examples()
    rng = random.Random(seed)
    rng.shuffle(rows)

    total_needed = train_n + val_n + test_n
    if total_needed > len(rows):
        raise ValueError(f"Need {total_needed} examples but only {len(rows)} available")

    def to_example(row):
        return dspy.Example(
            question=row["question"],
            answer=row["answer"],
        ).with_inputs("question")

    train = [to_example(r) for r in rows[:train_n]]
    val = [to_example(r) for r in rows[train_n:train_n + val_n]]
    test = [to_example(r) for r in rows[train_n + val_n:train_n + val_n + test_n]]
    return train, val, test


# ── DSPy Module ──────────────────────────────────────────────────────
class MGSMAnswer(dspy.Signature):
    """Solve a math problem step by step. Return the final numeric answer as an integer."""
    question: str = dspy.InputField(desc="The math problem to solve")
    reasoning: str = dspy.OutputField(desc="Step-by-step solution")
    answer: str = dspy.OutputField(desc="Final numeric answer (integer only)")


class MGSMSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(MGSMAnswer)

    def forward(self, question: str):
        return self.predict(question=question)


# ── Metric ───────────────────────────────────────────────────────────
def normalize_numeric(ans: str) -> str:
    """Extract numeric answer, strip commas and trailing decimals."""
    ans = str(ans).strip().replace(",", "")
    match = re.search(r"-?\d+\.?\d*", ans)
    if match:
        ans = match.group(0)
    if "." in ans:
        ans = ans.rstrip("0").rstrip(".")
    return ans


def mgsm_metric(gold, pred, trace=None):
    predicted = normalize_numeric(pred.answer)
    expected = normalize_numeric(gold.answer)
    return 1.0 if predicted == expected else 0.0


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
        temperature=0.6,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=lm)

    # Load data
    print(f"Loading MGSM splits: train={args.train_size}, val={args.val_size}, test={args.test_size}")
    trainset, valset, testset = load_mgsm_splits(
        args.train_size, args.val_size, args.test_size, seed=args.seed
    )

    student = MGSMSolver()

    # ── Baseline evaluation ──────────────────────────────────────────
    print("\n========== Baseline (no optimization) ==========")
    baseline_eval = dspy.Evaluate(
        devset=testset, metric=mgsm_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
        provide_traceback=True, max_errors=len(testset),
    )
    baseline_result = baseline_eval(student)
    baseline_score = float(baseline_result)
    print(f"Baseline accuracy: {baseline_score:.4f}")

    # ── MIPROv2 optimization ─────────────────────────────────────────
    print(f"\n========== MIPROv2 optimization (auto={args.auto}) ==========")
    t0 = time.time()

    mipro = dspy.MIPROv2(
        metric=mgsm_metric,
        auto=args.auto,
        num_threads=args.num_threads,
        track_stats=True,
        log_dir=str(out_dir / "mipro_logs"),
        seed=args.seed,
    )

    optimized = mipro.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )
    opt_time = time.time() - t0
    print(f"Optimization took {opt_time:.1f}s")

    # ── Post-optimization evaluation ─────────────────────────────────
    print("\n========== Optimized evaluation ==========")
    opt_eval = dspy.Evaluate(
        devset=testset, metric=mgsm_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
        provide_traceback=True, max_errors=len(testset),
    )
    opt_result = opt_eval(optimized)
    opt_score = float(opt_result)
    print(f"Optimized accuracy: {opt_score:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "method": "DSPy-MIPROv2",
        "model": args.model,
        "task": "mgsm",
        "auto": args.auto,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "baseline_accuracy": baseline_score,
        "optimized_accuracy": opt_score,
        "optimization_time_s": round(opt_time, 1),
        "seed": args.seed,
    }

    if hasattr(optimized, "trial_logs"):
        full_eval_scores = [
            v.get("full_eval_score")
            for v in optimized.trial_logs.values()
            if "full_eval_score" in v
        ]
        results["num_full_val_evals"] = len(full_eval_scores)
        results["best_val_score"] = max(full_eval_scores) if full_eval_scores else None
    if hasattr(optimized, "total_calls"):
        results["total_metric_calls"] = optimized.total_calls
    if hasattr(optimized, "candidate_programs"):
        results["num_candidates_explored"] = len(optimized.candidate_programs)

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
