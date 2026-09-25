"""Offline release regressions: no model calls, downloads, or GPU required."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent_module
import logic
import task_mgsm
from run_config import apply_run_config_to_env, load_run_config, redacted_config


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ConfigTests(unittest.TestCase):
    def test_config_merges_defaults_without_mutating_them(self):
        config = load_run_config(str(ROOT / "configs/experiment.yaml"))
        self.assertEqual(config["model"]["infer_model"], "qwen3-local")
        config["inner_loop"]["seed"] = 999
        self.assertEqual(load_run_config(None)["inner_loop"]["seed"], 7)

    def test_environment_key_takes_precedence(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-env-key"}, clear=True):
            apply_run_config_to_env({"model": {"api_key": "test-yaml-key"}})
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-env-key")

    def test_yaml_key_remains_supported(self):
        with patch.dict(os.environ, {}, clear=True):
            apply_run_config_to_env({"model": {"api_key": "test-yaml-key"}})
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-yaml-key")

    def test_single_server_clears_previous_pool(self):
        with patch.dict(os.environ, {}, clear=True):
            apply_run_config_to_env({"model": {"base_urls": ["http://one/v1", "http://two/v1"]}})
            self.assertIn("OPENAI_BASE_URLS", os.environ)
            apply_run_config_to_env({"model": {"base_url": "http://single/v1"}})
            self.assertNotIn("OPENAI_BASE_URLS", os.environ)

    def test_log_redaction_preserves_limits_and_original(self):
        config = {"model": {"api_key": "test-secret", "base_urls": [
            "https://user:password@example.test/v1?api_key=private"]},
            "token_management": {"max_input_tokens": 4096}}
        sanitized = redacted_config(config)
        output = json.dumps(sanitized)
        for sensitive in ("test-secret", "password", "private"):
            self.assertNotIn(sensitive, output)
        self.assertEqual(sanitized["token_management"]["max_input_tokens"], 4096)
        self.assertEqual(config["model"]["api_key"], "test-secret")

    def test_help_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "src/main.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)


class EvaluationTests(unittest.TestCase):
    def make_agent(self, task="mgsm", **kwargs):
        return agent_module.Agent(task_name=task,
            goal_prompt_path=str(ROOT / "src/goal_prompt.md"), **kwargs)

    def test_missing_optional_key_uses_local_placeholder(self):
        with patch.dict(os.environ, {}, clear=True):
            agent = self.make_agent()
            self.assertEqual(agent.client.api_key, "EMPTY")
            for client in agent._clients:
                client.close()

    def test_empty_legacy_key_uses_local_placeholder(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            key = Path(temp) / "key.env"
            key.touch()
            agent = self.make_agent(key_path=str(key))
            self.assertEqual(agent.client.api_key, "EMPTY")
            for client in agent._clients:
                client.close()

    def test_mgsm_runner_evaluates_and_creates_results(self):
        # Exercises the actual task.evaluate call, including max_workers and disk output.
        example = {"inputs": "What is 2 + 2?", "targets": "4", "lang": "en"}
        with tempfile.TemporaryDirectory() as temp, working_directory(temp), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(agent_module.logic, "store_all_logic"), \
             patch.object(agent_module, "best_eval_acc", 0.0), \
             patch.object(task_mgsm, "last_test_acc", 0.0), \
             patch.object(task_mgsm, "get_all_examples", return_value=[example] * 129), \
             patch.object(task_mgsm, "bootstrap_confidence_interval", return_value="CI"), \
             contextlib.redirect_stdout(io.StringIO()):
            agent = self.make_agent(max_workers=1)
            agent.selected_examples["evaluate"] = {"examples": [example]}
            report = agent_module.action_evaluate_on_task(
                agent, agent.goal_task, lambda task: {"answer": "4"})
            self.assertIn("Test Accuracy 1.0", report)
            self.assertTrue(Path("results/mgsm_1.0.txt").is_file())
            for client in agent._clients:
                client.close()

    def test_logic_snapshot_does_not_copy_credentials(self):
        with tempfile.TemporaryDirectory() as temp, working_directory(temp), \
             contextlib.redirect_stdout(io.StringIO()):
            Path("key.env").write_text("test-private-value")
            Path("goal_prompt.md").write_text("test goal")
            Path("snapshot").mkdir()
            logic.merge_and_clean("snapshot")
            self.assertFalse(Path("snapshot/key.env").exists())
            self.assertTrue(Path("snapshot/goal_prompt.md").exists())

    def test_all_registered_task_data_loads(self):
        self.assertEqual(set(agent_module.TASK_MODULES), {"mmlu", "mgsm", "ifeval"})
        with working_directory(ROOT):
            self.assertGreater(len(agent_module.task_mmlu._load_mmlu_examples()), 928)
            self.assertGreater(len(task_mgsm.get_all_examples()), 928)
            self.assertTrue(agent_module.task_ifeval._load_ifeval_examples("train"))
            self.assertTrue(agent_module.task_ifeval._load_ifeval_examples("test"))

    def test_bundled_data_matches_manifest(self):
        data = ROOT / "datasets"
        for relative, entry in json.loads((data / "manifest.json").read_text()).items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((data / relative).read_bytes()).hexdigest(),
                                 entry["sha256"])


class ServingTests(unittest.TestCase):
    def test_unknown_model_fails_before_launch(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/serve_llm.sh")],
            env={**os.environ, "MODEL": "invalid-model"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown MODEL", result.stderr)

    def test_qwen_aliases_use_same_server_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            interpreter = Path(temp) / "python3"
            interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            interpreter.chmod(0o755)
            commands = []
            for model in ("qwen3", "qwen3-4b-2507"):
                result = subprocess.run(["bash", str(ROOT / "scripts/serve_llm.sh")],
                    env={"PATH": temp + os.pathsep + os.environ["PATH"], "MODEL": model},
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands.append(result.stdout.splitlines()[1:])
            self.assertEqual(commands[0], commands[1])
            self.assertIn("hermes", commands[0])
            self.assertIn("127.0.0.1", commands[0])
            self.assertIn("qwen3-local", commands[0])

    def test_busy_gpu_refuses_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            query = Path(temp) / "nvidia-smi"
            query.write_text("#!/bin/sh\necho 12345\n")
            query.chmod(0o755)
            result = subprocess.run(["bash", str(ROOT / "scripts/serve_all.sh")],
                env={**os.environ, "PATH": temp + os.pathsep + os.environ["PATH"],
                     "NUM_SERVERS": "1"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("already has compute processes", result.stderr)
            self.assertNotIn("PID", result.stdout)


if __name__ == "__main__":
    unittest.main()
