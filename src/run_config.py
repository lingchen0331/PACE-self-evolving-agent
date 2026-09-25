import copy
import os
from typing import Any, Dict

DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "task": "mmlu",
    "runs": 1,
    "goal_prompt_path": "src/goal_prompt.md",
    "key_path": None,
    "max_outer_evolve_steps": 10,
    "disable_tool_calls": True,
    "model": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": None,
        "infer_model": "qwen3-local",
        "optimizer_model": "qwen3-local",
        "outer_loop_model": "qwen3-local",
    },
    "inner_loop": {
        "iterations": 4,
        "train_size": 32,
        "valid_size": 32,
        "candidates_per_iter": 8,
        "seed": 7,
        "rotate_seed_each_call": True,
    },
    "evolution_credit": {
        "prompt_saturation_epsilon": 0.01,
        "fallback_min_prompt_update_rounds": 2,
    },
    "token_management": {
        "max_input_tokens": 40960,
        "max_output_tokens": 4000,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_run_config(path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_RUN_CONFIG)
    if path is None:
        return config

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    lower = path.lower()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if lower.endswith(".yaml") or lower.endswith(".yml"):
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML. Install with `pip install pyyaml`.") from e
        user_cfg = yaml.safe_load(raw)
    else:
        raise ValueError("Config file must be .yaml/.yml")

    if user_cfg is None:
        user_cfg = {}
    if not isinstance(user_cfg, dict):
        raise ValueError("Top-level config must be a mapping/object.")

    return _deep_update(config, user_cfg)


def apply_run_config_to_env(config: Dict[str, Any]) -> None:
    model_cfg = config.get("model", {}) or {}
    if model_cfg.get("base_url"):
        os.environ["OPENAI_BASE_URL"] = str(model_cfg["base_url"])
    base_urls = model_cfg.get("base_urls")
    if isinstance(base_urls, list) and base_urls:
        os.environ["OPENAI_BASE_URLS"] = ",".join(str(u) for u in base_urls)
    elif model_cfg.get("base_url"):
        os.environ.pop("OPENAI_BASE_URLS", None)
    if model_cfg.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(model_cfg["api_key"])
    if model_cfg.get("infer_model"):
        os.environ["AGENT_MODEL_NAME"] = str(model_cfg["infer_model"])
    if model_cfg.get("optimizer_model"):
        os.environ["OPTIMIZER_MODEL_NAME"] = str(model_cfg["optimizer_model"])
    if model_cfg.get("outer_loop_model"):
        os.environ["OUTER_LOOP_MODEL_NAME"] = str(model_cfg["outer_loop_model"])

    disable_tools = config.get("disable_tool_calls", False)
    os.environ["DISABLE_TOOL_CALLS"] = "1" if bool(disable_tools) else "0"

    token_cfg = config.get("token_management", {}) or {}
    if token_cfg.get("max_input_tokens") is not None:
        os.environ["MAX_INPUT_TOKENS"] = str(token_cfg["max_input_tokens"])
    if token_cfg.get("max_output_tokens") is not None:
        os.environ["MAX_OUTPUT_TOKENS"] = str(token_cfg["max_output_tokens"])


def redacted_config(config):
    """Return a logging-safe copy without credential fields or URL credentials."""
    from urllib.parse import urlsplit, urlunsplit

    if isinstance(config, dict):
        return {
            key: "[REDACTED]" if any(part in key.lower() for part in
                ("api_key", "token", "password", "secret")) and not isinstance(value, (dict, int, float))
            else redacted_config(value)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [redacted_config(value) for value in config]
    if isinstance(config, str) and "://" in config:
        url = urlsplit(config)
        return urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    return config
