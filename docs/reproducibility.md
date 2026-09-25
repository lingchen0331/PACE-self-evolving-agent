# Reproducibility and release coverage

This document distinguishes the current implementation from the complete experiment suite described in the [paper](https://arxiv.org/abs/2605.23019). Cleanup changes preserve the existing research scoring and evolution algorithms, except for the MGSM evaluation call compatibility fix.

## Available entry points

`src/main.py` registers MMLU, MGSM, and IFEval. HotpotQA and τ-bench PACE implementations are absent. The legacy `task_gpqa.py` is not integrated with the runner. Baseline/comparison methods, ablation launchers, and comparison plotting utilities are intentionally excluded from this PACE-focused release. Restore the original PACE experiment configurations and missing PACE task implementations before claiming reproduction of all PACE results.

The example configuration uses one Qwen3 server, a 20-step outer budget, and smaller context limits for setup convenience. It replaces a machine-specific two-server Ministral configuration. It is not evidence of the configuration used to produce the paper's tables.

## Evaluation details that affect interpretation

- **IFEval:** `src/task_ifeval.py` uses custom heuristic checkers. Unknown instruction IDs are accepted, and the language checker only tests for nonempty text. The reported `strict` value is an all-checks-pass score; `loose` is a fraction of passed instructions, not the official IFEval loose metric. These scores must not be presented as verified official IFEval results. See the [official evaluator](https://github.com/google-research/google-research/tree/master/instruction_following_eval). Replacing the scorer requires rerunning experiments.
- **Data selection:** MMLU and MGSM shuffle with seed 0, use the first 128 examples as the evolution pool, and positions 128–927 for full evaluation. IFEval reads separate bundled train/test files and uses the first 128 shuffled train examples for its outer evaluation pool. Inner prompt search uses configured sample sizes; these are not complete benchmark training splits.
- **Feedback and selection:** Evaluation hooks can expose full-test scores during evolution. The controller records the returned score in its credit state and uses it when deciding whether to snapshot improvements. A strict held-out final-test protocol requires separating model selection from final reporting and rerunning experiments; this cleanup does not retrofit that protocol.
- **Conditional full evaluation:** Task evaluators may skip full evaluation below a validation threshold and return zero or cached test scores. Therefore `max_outer_evolve_steps: 0` requests the existing baseline evaluation path, not an unconditional full-benchmark score.
- **Randomness:** Inner prompt search has a seed, but several outer evaluation paths reseed using wall-clock time. One YAML seed does not make an entire run deterministic. Multiple runs share some module-level evaluation state; use separate processes for independent trials.
- **Concurrency:** The worker configuration is not global. Some prompt-search and full-test paths use hard-coded worker counts. Reduce these in the corresponding task module if your server cannot sustain the load.
- **Validation scope:** Native `action_adjust_logic` calls targeting `agent_module.solver` pass through the A/B gate. Other edit targets and arbitrary code execution are not comprehensively sandboxed or subjected to that same gate. Prompt-only fallback does not perform structural evolution.
- **Cost:** The outer step count is an evolution budget. The implementation does not enforce a universal per-query token or model-call budget across generated solvers.
- **Outputs:** Source snapshots are best-effort introspection artifacts, not complete executable checkpoints. Keep configurations and logs alongside them.

## What remains to reproduce all PACE experiments

1. Supply the original HotpotQA PACE and τ-bench implementations and their original PACE experiment configurations.
2. Resolve and report the IFEval scorer and final-test protocol differences above.
3. Supply exact environment locks, model revisions, data preparation/split scripts, seeds, and run-level results for the published tables.
4. Confirm upstream attribution notices and redistribution permissions for bundled benchmark snapshots; their exact upstream revisions are not recorded in the original checkout.
5. Run end-to-end GPU experiments with the release configuration. CPU checks validate setup and regression behavior only.

The existing MIT license is retained. Acceptance venue metadata should be updated when the final citation is supplied.
