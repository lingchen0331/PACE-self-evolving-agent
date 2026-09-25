# PACE: Two-Timescale Self-Evolution for Small Language Model Agents

[![Paper](https://img.shields.io/badge/arXiv-2605.23019-b31b1b.svg)](https://arxiv.org/abs/2605.23019)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Research code for **PACE (Prompt And Control Logic Evolution)**. PACE adapts the prompts and control logic around a frozen small language model, without updating model weights. It refines prompts first, considers structural changes after prompt gains saturate, and evaluates candidate solver changes before accepting them.

**Paper:** [PACE: Two-Timescale Self-Evolution for Small Language Model Agents](https://arxiv.org/abs/2605.23019)
Chen Ling, Pei Chen, Albert Guan, Jiaming Qu, Shayan Ali Akbar, Madhu Gopinathan, and Erwin Cornejo.

## Quick start

Run commands from the repository root. Use **Python 3.10 or newer** for the client (the setup helper uses Python 3.12). Model serving needs a separate compatible GPU environment; the supplied serving scripts target Linux with NVIDIA GPUs. An existing OpenAI-compatible inference endpoint also works.

```bash
git clone https://github.com/lingchen0331/PACE-self-evolving-agent.git
cd PACE-self-evolving-agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The client dependencies do not install vLLM or download model weights.

### 1. Start a model server

On a GPU machine, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and create a separate serving environment:

```bash
bash scripts/install_vllm.sh
source .venv-serving/bin/activate
python model_downloader.py --target qwen3
MODEL=qwen3 bash scripts/serve_llm.sh
```

This downloads `Qwen/Qwen3-4B-Instruct-2507` and serves it as `qwen3-local` at `http://127.0.0.1:8000/v1`, matching the example configuration. Model downloads can be large. The installer does not download checkpoints automatically or replace a separately installed PyTorch with a hard-coded CUDA build. Consult the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for your hardware.

Other serving options:

| `MODEL` | Local checkpoint directory | Served model name |
| --- | --- | --- |
| `qwen3` (also `qwen3-4b-2507`) | `local_qwen3_model/` | `qwen3-local` |
| `qwen3.5` | `local_qwen35_model/` | `qwen3.5-local` |
| `ministral-14b` | `local_ministral_model/` | `ministral-14b-local` |

Download Qwen3.5 with `python model_downloader.py --target qwen3.5`. For other checkpoints, use `--model-name <model-id> --output-dir <directory>`. Change all three model names in the YAML when switching models. `MODEL_PATH`, `SERVED_MODEL_NAME`, `SERVE_PORT`, `MAX_MODEL_LEN`, `TP_SIZE`, and `GPU_MEMORY_UTILIZATION` can override serving settings. The default context length is 32,768 tokens; keep the client input and output limits within the server's capacity.

### 2. Configure the experiment

[configs/experiment.yaml](configs/experiment.yaml) provides a single-server MMLU example. Copy it for local experiments:

```bash
cp configs/experiment.yaml configs/experiment.local.yaml
```

| Setting | Purpose |
| --- | --- |
| `task` | `mmlu`, `mgsm`, or `ifeval` |
| `model.base_url` | Inference endpoint; use `model.base_urls` for round-robin requests |
| `model.infer_model` | Solver model name |
| `model.optimizer_model` / `model.outer_loop_model` | Prompt proposer / controller model names |
| `max_outer_evolve_steps` | Outer evolution budget; `0` requests baseline evaluation |
| `inner_loop` | Prompt search iterations, sample sizes, candidates, and seed |
| `evolution_credit.prompt_saturation_epsilon` | Threshold for prompt saturation |
| `evolution_credit.ab_gate_delta_threshold` | Required score improvement for gated solver edits |
| `disable_tool_calls` | `false` for structural evolution; `true` uses prompt-only fallback |
| `token_management` | Input/output request limits |
| `concurrency.max_workers` | Worker limit for supported task evaluation paths |

Use the same frozen model for all three roles for the paper's self-evolution setting. The bundled YAML is a starting configuration, not a complete reproduction manifest. Some inner and full-test evaluation paths retain their own worker limits; see the reproducibility notes.

For authenticated endpoints, export `OPENAI_API_KEY` in your shell. It takes precedence over an optional YAML key. Local unauthenticated vLLM uses `EMPTY` automatically. Do not commit credentials; `*.env` and `configs/*.local.yaml` are ignored.

### 3. Run PACE

In another terminal, activate the client environment:

```bash
source .venv/bin/activate
mkdir -p logs
bash scripts/run_experiment.sh configs/experiment.local.yaml 2>&1 | tee logs/experiment.log
```

The runner creates `results/` automatically. Evaluation reports are written there when the task's score gate permits; improved agents can also produce source snapshots in root-level `<task>_<score>/` directories. Logs, checkpoints, and generated artifacts are ignored by Git. Snapshots contain introspected source and are not guaranteed to be standalone runnable packages.

**Execution environment:** PACE executes model-generated Python and shell commands in its own process environment. The validation gate measures solver quality; it is not an operating-system sandbox. Run evolution in a disposable container or dedicated machine without personal credentials or unrelated workloads.

## Serving utilities

`serve_all.sh` launches one server per idle GPU and refuses occupied GPUs or ports. Use `MODEL` and `NUM_SERVERS` to configure it, then set `model.base_urls` in the experiment YAML to the matching endpoints. `lb_proxy.py` is an optional local round-robin proxy.

## Repository layout

```text
configs/                  Example run configuration
src/agent_module.py       Controller, tool dispatch, prompt credit, solver validation
src/task_*.py             Task loaders, solvers, prompt search, evaluation
src/run_config.py         Configuration loading and environment setup
src/goal_prompt.md        Controller instructions
src/logic.py              Source introspection and evolved-logic snapshots
scripts/                  Model serving and experiment launch utilities
datasets/                 Bundled benchmark files and provenance notes
tests/                    CPU-only regression and integration tests
docs/reproducibility.md    Release coverage and experiment limitations
```

## Development checks

No GPU or live model endpoint is needed:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts model_downloader.py
for script in scripts/*.sh; do bash -n "$script"; done
```

These checks validate code paths and setup behavior; they do not establish benchmark performance. The dependency files are not a lockfile for the original experiments. Record the installed versions, checkpoint revision, configuration, and logs for each run.

## Citation

```bibtex
@misc{ling2026pace,
  title={PACE: Two-Timescale Self-Evolution for Small Language Model Agents},
  author={Chen Ling and Pei Chen and Albert Guan and Jiaming Qu and
          Shayan Ali Akbar and Madhu Gopinathan and Erwin Cornejo},
  year={2026},
  eprint={2605.23019},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.23019}
}
```

## License and acknowledgments

Project code is distributed under the [MIT License](LICENSE). Benchmark datasets and model checkpoints retain their respective upstream terms. Task evaluation code was adapted from [ADAS](https://github.com/ShengranHu/ADAS), as acknowledged in the original repository. See [dataset provenance](datasets/readme.md) for benchmark sources.
