import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelPreset:
    repo_id: str
    output_dir: str
    served_model_name: str


MODEL_PRESETS = {
    "qwen3": ModelPreset(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir="./local_qwen3_model",
        served_model_name="qwen3-local",
    ),
    "qwen3.5": ModelPreset(
        repo_id="Qwen/Qwen3.5-9B",
        output_dir="./local_qwen35_model",
        served_model_name="qwen3.5-local",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the repo's default local LLM checkpoints or a specific Hugging Face model."
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(MODEL_PRESETS),
        help="Named preset to download. Repeat to select multiple presets. Defaults to all presets.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME"),
        help="Custom Hugging Face model id to download instead of presets.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MODEL_OUTPUT_DIR"),
        help="Directory where a custom checkpoint and tokenizer will be saved.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip a short tokenizer smoke test after each model is present locally.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination already looks populated.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print the available preset targets and exit.",
    )
    return parser.parse_args()


def _looks_downloaded(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    has_config = (output_dir / "config.json").is_file()
    has_tokenizer = (output_dir / "tokenizer.json").is_file() or (output_dir / "tokenizer_config.json").is_file()
    has_weights = any(output_dir.glob("*.safetensors")) or (output_dir / "model.safetensors.index.json").is_file()
    return has_config and has_tokenizer and has_weights


def _smoke_test_tokenizer(output_dir: Path) -> None:
    from transformers import AutoTokenizer

    print("Loading tokenizer for a quick smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    print("Tokenizer smoke test:", inputs[:120])


def _download_one(repo_id: str, output_dir: str, served_model_name: str | None, force: bool, skip_smoke_test: bool) -> None:
    output_path = Path(output_dir)
    label = f"{repo_id} -> {output_dir}"
    if served_model_name:
        label = f"{label} (served as {served_model_name})"

    if _looks_downloaded(output_path) and not force:
        print(f"Skipping {label}; files already exist.")
    else:
        print(f"Downloading {label}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
        )

    if not skip_smoke_test:
        _smoke_test_tokenizer(output_path)

    print(f"Ready: {output_dir}")


def _default_targets() -> list[str]:
    env_targets = os.environ.get("MODEL_TARGETS", "").strip()
    if not env_targets:
        return list(MODEL_PRESETS.keys())
    return [target.strip() for target in env_targets.split(",") if target.strip()]


def main() -> None:
    args = parse_args()

    if args.list_targets:
        for name, preset in MODEL_PRESETS.items():
            print(f"{name}: {preset.repo_id} -> {preset.output_dir} (served as {preset.served_model_name})")
        return

    if args.model_name:
        output_dir = args.output_dir
        if not output_dir:
            raise ValueError("--output-dir is required when using --model-name.")
        _download_one(
            repo_id=args.model_name,
            output_dir=output_dir,
            served_model_name=None,
            force=args.force,
            skip_smoke_test=args.skip_smoke_test,
        )
        return

    targets = args.target or _default_targets()
    unknown_targets = [target for target in targets if target not in MODEL_PRESETS]
    if unknown_targets:
        raise ValueError(f"Unknown targets: {', '.join(unknown_targets)}")

    for target in targets:
        preset = MODEL_PRESETS[target]
        _download_one(
            repo_id=preset.repo_id,
            output_dir=preset.output_dir,
            served_model_name=preset.served_model_name,
            force=args.force,
            skip_smoke_test=args.skip_smoke_test,
        )


if __name__ == "__main__":
    main()
