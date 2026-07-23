#!/usr/bin/env python3
"""
DSPy GEPA baseline for HotpotQA (distractor setting).

Usage:
    source .venv/bin/activate
    python scripts/baseline_dspy_gepa_hotpotqa.py [--model qwen3-local] [--auto medium]

Requires vLLM servers on ports 8000/8001 (see scripts/serve_all.sh).
Expects datasets/hotpot_qa/ saved via datasets.save_to_disk().
"""
import argparse
import collections
import itertools
import json
import random
import re
import string
import threading
import time
from pathlib import Path

import dspy
from datasets import load_from_disk


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
    p = argparse.ArgumentParser(description="DSPy GEPA baseline on HotpotQA")
    p.add_argument("--model", default="qwen3-local")
    p.add_argument("--base-urls", nargs="+",
                   default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"])
    p.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                   help="GEPA budget preset")
    p.add_argument("--train-size", type=int, default=200,
                   help="Number of training examples for GEPA")
    p.add_argument("--val-size", type=int, default=100,
                   help="Number of validation examples for GEPA")
    p.add_argument("--test-size", type=int, default=500,
                   help="Number of test examples from the validation split")
    p.add_argument("--num-threads", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="Max output tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results/dspy_gepa_hotpotqa")
    return p.parse_args()


# ── Answer normalization (SQuAD-style, standard for HotpotQA) ────────
def normalize_answer(s: str) -> str:
    """Lower text, remove articles, punctuation, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Data ─────────────────────────────────────────────────────────────
def format_context(context_dict: dict) -> str:
    """Format HotpotQA context paragraphs into a readable string."""
    titles = context_dict["title"]
    sentences_list = context_dict["sentences"]
    parts = []
    for title, sentences in zip(titles, sentences_list):
        para = "".join(sentences).strip()
        parts.append(f"[{title}]\n{para}")
    return "\n\n".join(parts)


def load_hotpotqa_splits(train_n, val_n, test_n, seed=0):
    ds = load_from_disk("datasets/hotpot_qa")
    train_raw = list(ds["train"])
    val_raw = list(ds["validation"])

    rng = random.Random(seed)
    rng.shuffle(train_raw)
    rng.shuffle(val_raw)

    # train split -> GEPA train + val; validation split -> test
    total_train_needed = train_n + val_n
    if total_train_needed > len(train_raw):
        print(f"Warning: requested {total_train_needed} train+val but only {len(train_raw)} available, adjusting.")
        train_n = min(train_n, len(train_raw) * 2 // 3)
        val_n = min(val_n, len(train_raw) - train_n)
    test_n = min(test_n, len(val_raw))

    def to_example(row):
        context_str = format_context(row["context"])
        return dspy.Example(
            question=row["question"],
            context=context_str,
            answer=row["answer"],
        ).with_inputs("question", "context")

    train = [to_example(r) for r in train_raw[:train_n]]
    val = [to_example(r) for r in train_raw[train_n:train_n + val_n]]
    test = [to_example(r) for r in val_raw[:test_n]]
    return train, val, test


# ── DSPy Module ──────────────────────────────────────────────────────
class HotpotQAAnswer(dspy.Signature):
    """Answer a multi-hop question using the provided context paragraphs. Give a short, precise answer."""
    context: str = dspy.InputField(desc="Supporting context paragraphs")
    question: str = dspy.InputField(desc="A multi-hop question requiring reasoning over multiple paragraphs")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning connecting facts from different paragraphs")
    answer: str = dspy.OutputField(desc="A short, precise answer (a few words)")


class HotpotQASolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(HotpotQAAnswer)

    def forward(self, question: str, context: str):
        return self.predict(question=question, context=context)


# ── Metric ───────────────────────────────────────────────────────────
def hotpotqa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    predicted = str(getattr(pred, "answer", "") or "")
    expected = str(gold.answer)

    em = exact_match_score(predicted, expected)
    f1 = f1_score(predicted, expected)

    if pred_name is not None:
        if em == 1.0:
            feedback = f"Correct! The answer is '{expected}'."
        else:
            feedback = (
                f"Incorrect. Predicted '{predicted}' but expected '{expected}'. "
                f"EM={em:.0f}, F1={f1:.2f}. "
                f"The answer must exactly match the gold answer after normalization."
            )
        return dspy.Prediction(score=em, feedback=feedback)

    return em


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
    print(f"Loading HotpotQA splits: train={args.train_size}, val={args.val_size}, test={args.test_size}")
    trainset, valset, testset = load_hotpotqa_splits(
        args.train_size, args.val_size, args.test_size, seed=args.seed
    )
    print(f"  Actual sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    student = HotpotQASolver()

    # ── Baseline evaluation ──────────────────────────────────────────
    print("\n========== Baseline (no optimization) ==========")
    baseline_eval = dspy.Evaluate(
        devset=testset, metric=hotpotqa_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
        provide_traceback=True, max_errors=len(testset),
    )
    baseline_result = baseline_eval(student)
    baseline_em = float(baseline_result)
    print(f"Baseline EM: {baseline_em:.4f}")

    # Also compute F1 on baseline
    baseline_f1_scores = []
    with dspy.context(lm=lm):
        for ex in testset:
            try:
                pred = student(question=ex.question, context=ex.context)
                baseline_f1_scores.append(f1_score(str(pred.answer), str(ex.answer)))
            except Exception:
                baseline_f1_scores.append(0.0)
    baseline_f1 = sum(baseline_f1_scores) / len(baseline_f1_scores) if baseline_f1_scores else 0.0
    print(f"Baseline F1: {baseline_f1:.4f}")

    # ── GEPA optimization ────────────────────────────────────────────
    print(f"\n========== GEPA optimization (auto={args.auto}) ==========")
    t0 = time.time()

    gepa = dspy.GEPA(
        metric=hotpotqa_metric,
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
        devset=testset, metric=hotpotqa_metric, num_threads=args.num_threads,
        display_progress=True, display_table=0,
        provide_traceback=True, max_errors=len(testset),
    )
    opt_result = opt_eval(optimized)
    opt_em = float(opt_result)
    print(f"Optimized EM: {opt_em:.4f}")

    # Also compute F1 on optimized
    opt_f1_scores = []
    with dspy.context(lm=lm):
        for ex in testset:
            try:
                pred = optimized(question=ex.question, context=ex.context)
                opt_f1_scores.append(f1_score(str(pred.answer), str(ex.answer)))
            except Exception:
                opt_f1_scores.append(0.0)
    opt_f1 = sum(opt_f1_scores) / len(opt_f1_scores) if opt_f1_scores else 0.0
    print(f"Optimized F1: {opt_f1:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "method": "DSPy-GEPA",
        "model": args.model,
        "task": "hotpotqa",
        "auto": args.auto,
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "baseline_em": baseline_em,
        "baseline_f1": round(baseline_f1, 4),
        "optimized_em": opt_em,
        "optimized_f1": round(opt_f1, 4),
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
    print(f"  Baseline:  EM={baseline_em:.4f}  F1={baseline_f1:.4f}")
    print(f"  Optimized: EM={opt_em:.4f}  F1={opt_f1:.4f}")
    print(f"  Delta:     EM={opt_em - baseline_em:+.4f}  F1={opt_f1 - baseline_f1:+.4f}")


if __name__ == "__main__":
    main()
