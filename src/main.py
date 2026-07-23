import argparse
import functools
from pathlib import Path

from agent_module import Agent
from agent_module import action_evaluate_on_task
from agent_module import solver
from run_config import apply_run_config_to_env, load_run_config


def print_config_header(config_path: str) -> None:
    resolved_path = Path(config_path).expanduser().resolve()
    print("========== Run Config: ==========")
    print(f"path={resolved_path}")
    try:
        print(resolved_path.read_text(encoding="utf-8").rstrip())
    except FileNotFoundError:
        print(f"(config file not found: {resolved_path})")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML run config.")
    args = parser.parse_args()

    print_config_header(args.config)
    cfg = load_run_config(args.config)
    apply_run_config_to_env(cfg)

    key_path = cfg["key_path"]
    goal_prompt_path = cfg["goal_prompt_path"]
    task_name = cfg["task"]
    runs = int(cfg.get("runs", 1))
    max_outer_evolve_steps = int(cfg.get("max_outer_evolve_steps", 10))
    inner_loop_defaults = cfg.get("inner_loop", {})
    evolution_credit_cfg = cfg.get("evolution_credit", {})
    token_management_cfg = cfg.get("token_management", {})
    concurrency_cfg = cfg.get("concurrency", {})
    prompt_saturation_epsilon = float(evolution_credit_cfg.get("prompt_saturation_epsilon", 0.01))
    fallback_min_prompt_update_rounds = int(evolution_credit_cfg.get("fallback_min_prompt_update_rounds", 2))
    ab_gate_delta_threshold = float(evolution_credit_cfg.get("ab_gate_delta_threshold", 0.02))
    max_input_tokens = int(token_management_cfg.get("max_input_tokens", 128000))
    max_output_tokens = int(token_management_cfg.get("max_output_tokens", 4000))
    max_workers = int(concurrency_cfg.get("max_workers", 48))

    for _ in range(runs):
        self_evolving_agent = Agent(
            goal_prompt_path=goal_prompt_path,
            key_path=key_path,
            task_name=task_name,
            max_outer_evolve_steps=max_outer_evolve_steps,
            inner_loop_defaults=inner_loop_defaults,
            prompt_saturation_epsilon=prompt_saturation_epsilon,
            fallback_min_prompt_update_rounds=fallback_min_prompt_update_rounds,
            ab_gate_delta_threshold=ab_gate_delta_threshold,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_workers=max_workers,
        )
        self_evolving_agent.reinit()
        if max_outer_evolve_steps <= 0:
            print("========== Outer Loop Skipped ==========\nmax_outer_evolve_steps <= 0, running baseline evaluation only.\n")
            baseline_result = action_evaluate_on_task(
                self_evolving_agent,
                self_evolving_agent.goal_task,
                functools.partial(solver, self_evolving_agent),
            )
            print(f"========== Baseline Evaluation Result: ==========\n{baseline_result}")
            continue
        self_evolving_agent.evolve()
