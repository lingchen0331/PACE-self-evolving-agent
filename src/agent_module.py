import os
import io
import re
import ast
import sys
import json
import random
import typing
import inspect
import functools
import itertools
import traceback
import importlib
import subprocess
import contextlib
import collections
import time
try:
    import openai
except Exception:
    openai = None
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import logic
import task_mmlu as task_mmlu
import task_mgsm as task_mgsm
import task_ifeval as task_ifeval


action_counter = collections.defaultdict(int)
best_eval_acc = 0.0
logic_dump_improvement_epsilon = 0.02
TASK_MODULES = {
    "mmlu": task_mmlu,
    "mgsm": task_mgsm,
    "ifeval": task_ifeval,
}


def _format_exception(prefix: str = "Error ") -> str:
    exception_stringio = io.StringIO()
    traceback.print_exc(file=exception_stringio)
    result = prefix + exception_stringio.getvalue()
    exception_stringio.close()
    return result


def _extract_key_lines(text: str, prefixes: typing.Iterable[str], limit: int = 12) -> typing.List[str]:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in prefixes):
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _summarize_tool_result(tool_name: str, result: typing.Any) -> str:
    text = str(result or "").strip()
    if not text:
        return "(no output)"

    summary_prefixes = {
        "action_evaluate_on_task": (
            "Valid Accuracy:",
            "Valid Strict Accuracy:",
            "Valid Loose Accuracy:",
            "Test Accuracy",
            "Test Strict Accuracy:",
            "Reflection Summary:",
            "- Dominant failure modes:",
            "- Frequent failed instruction ids:",
            "- Evolution direction:",
            "Delta_U_P=",
            "Delta_U_C=",
            "PromptSaturated=",
            "LastEvalAcc=",
        ),
        "action_optimize_prompt_on_task": (
            "Inner Prompt Optimization Finished.",
            "Train samples:",
            "Candidates on Pareto front:",
            "Initial valid accuracy:",
            "Best valid accuracy:",
            "Estimated Delta_U_P:",
            "Best prompt config:",
            "Delta_U_P=",
        ),
        "action_get_evolution_credit": (
            "Delta_U_P=",
            "Delta_U_C=",
            "Epsilon=",
            "PromptSaturated=",
            "PendingStructureEval=",
            "LastEvalAcc=",
        ),
        "action_select_examples": (
            "Selected ",
            "Stored for future",
            "Preview only",
            "Top groups:",
            "Preview:",
        ),
    }

    if tool_name in summary_prefixes:
        lines = _extract_key_lines(text, summary_prefixes[tool_name])
        if lines:
            return "\n".join(lines)

    if tool_name == "action_display_analysis":
        return "\n".join(text.splitlines()[:20])

    if tool_name == "action_read_logic":
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return "(empty logic)"
        preview = lines[:6]
        if len(lines) > 6:
            preview.append(f"... ({len(lines)} lines total)")
        return "\n".join(preview)

    if tool_name == "action_call_json_format_llm":
        lines = text.splitlines()
        return "\n".join(lines[:8])

    lines = text.splitlines()
    if len(lines) <= 8 and len(text) <= 600:
        return text
    preview = lines[:8]
    preview.append("...")
    return "\n".join(preview)

class AgentBase:
    def execute_action(agent, *args, **kwargs):
        raise NotImplementedError("execute_action hasn't been implemented")
    
    def action_call_llm(agent, *args, **kwargs):
        raise NotImplementedError("action_call_llm hasn't been implemented")

def action_display_analysis(agent: AgentBase, analysis: str = "") -> str:
    """
    Summarize the latest evaluation failures into a structured taxonomy and
    combine with the agent's own analysis to produce actionable evolution
    guidance for the next outer-loop iteration.

    Args:
        agent: The current agent instance (provides credit_state and eval history).
        analysis: Free-text analysis or plan written by the agent.

    Returns:
        A structured string containing the failure taxonomy and next-step guidance.
    """
    print(f"========== Analysis:  ==========\n{analysis}", "\n\n")

    # ── pull last evaluation feedback stored by _summarize_evaluation_feedback ──
    reflection = agent.credit_state.get("last_failure_reflection", "")
    last_eval_acc = agent.credit_state.get("last_eval_acc")
    prompt_saturated = agent.credit_state.get("prompt_saturated", False)
    delta_u_p = float(agent.credit_state.get("last_prompt_delta", 0.0))
    delta_u_c = agent.credit_state.get("last_structure_delta")

    # ── parse the reflection summary produced by _summarize_evaluation_feedback ──
    taxonomy = {
        "extraction_runtime": 0,
        "format_constraint": 0,
        "reasoning_content": 0,
        "unclassified": 0,
    }
    total_incorrect = 0
    frequent_ids = "none"
    raw_metrics = "none parsed"

    if reflection:
        m = re.search(r"incorrect:\s*(\d+)", reflection)
        if m:
            total_incorrect = int(m.group(1))
        for key, pattern in [
            ("extraction_runtime", r"extraction/runtime=(\d+)"),
            ("format_constraint", r"format/constraint=(\d+)"),
            ("reasoning_content", r"reasoning/content=(\d+)"),
            ("unclassified", r"unclassified=(\d+)"),
        ]:
            m = re.search(pattern, reflection)
            if m:
                taxonomy[key] = int(m.group(1))
        m = re.search(r"Frequent failed instruction ids:\s*(.+)", reflection)
        if m:
            frequent_ids = m.group(1).strip()
        m = re.search(r"Metrics:\s*(\{.+?\})", reflection)
        if m:
            raw_metrics = m.group(1)

    # ── rank failure modes by severity ──
    ranked = sorted(taxonomy.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(k, v) for k, v in ranked if v > 0]

    # ── build targeted recommendations ──
    recommendations = []
    if taxonomy["extraction_runtime"] > 0:
        recommendations.append(
            f"[extraction/runtime × {taxonomy['extraction_runtime']}] "
            "Solver output is missing expected keys or crashing. "
            "Harden the solver's output parsing and add fallback extraction logic."
        )
    if taxonomy["format_constraint"] > 0:
        recommendations.append(
            f"[format/constraint × {taxonomy['format_constraint']}] "
            "Responses violate formatting or instruction-following constraints. "
            "Restate constraint rules more explicitly in prompt requirements, "
            "or add a post-generation validation step in the solver."
        )
    if taxonomy["reasoning_content"] > 0:
        recommendations.append(
            f"[reasoning/content × {taxonomy['reasoning_content']}] "
            "Answers are wrong despite correct formatting. "
            "Consider chain-of-thought decomposition, self-verification, "
            "or domain-specific reasoning scaffolds in the solver."
        )
    if taxonomy["unclassified"] > 0:
        recommendations.append(
            f"[unclassified × {taxonomy['unclassified']}] "
            "Some failures don't fit known categories. "
            "Inspect sample outputs manually via action_run_code to identify new patterns."
        )

    # ── evolution strategy hint ──
    if last_eval_acc is not None and last_eval_acc >= 0.9:
        strategy = (
            "Performance is strong (≥0.9). Prefer conservative, low-risk changes "
            "and verify regressions carefully."
        )
    elif prompt_saturated:
        strategy = (
            f"Prompt gains are saturated (Delta_U_P={delta_u_p:.4f} < ε). "
            "Structural edits (solver logic, post-processing, tool creation) "
            "are now the best path forward."
        )
    else:
        strategy = (
            f"Prompt gains are NOT saturated (Delta_U_P={delta_u_p:.4f}). "
            "Favor inner-loop prompt optimization before risky structural edits."
        )

    delta_u_c_str = f"{float(delta_u_c):.4f}" if delta_u_c is not None else "N/A"

    sections = [
        "=== Failure Taxonomy ===",
        f"Total incorrect: {total_incorrect}",
        f"Breakdown: {', '.join(f'{k}={v}' for k, v in ranked) if ranked else 'no failures parsed'}",
        f"Frequent failed IDs: {frequent_ids}",
        f"Metrics: {raw_metrics}",
        "",
        "=== Targeted Recommendations ===",
        *([r for r in recommendations] if recommendations else ["No specific failure patterns detected yet. Run action_evaluate_on_task first."]),
        "",
        "=== Evolution Strategy ===",
        strategy,
        f"Delta_U_P={delta_u_p:.4f}  Delta_U_C={delta_u_c_str}  "
        f"Last eval acc={last_eval_acc if last_eval_acc is not None else 'N/A'}",
        "",
        "=== Agent Analysis ===",
        analysis if analysis.strip() else "(no analysis provided)",
    ]

    return "\n".join(sections)

def action_environment_aware(agent: AgentBase):
    """
    Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.

    Returns:
        summary (str): Summary of available resources.
    """
    def summarize_items(items, header):
        summary = [header]
        for name, value in items:
            if not name.startswith('__'):
                if name in ['goal_prompt']:
                    summary.append(f"- {name} = Your {name}.")
                elif name in ['optimize_history', 'function_map', 'action_functions']:
                    summary.append(f"- {name} = The length of your {name} is {len(getattr(agent, name))}.")
                elif name == 'selected_examples':
                    selected = getattr(agent, name, {})
                    sizes = {
                        key: len(value.get("examples", []))
                        for key, value in selected.items()
                        if isinstance(value, dict)
                    }
                    summary.append(f"- {name} = Stored example selections by target split: {sizes}")
                elif name in ['logic']:
                    pass
                else:
                    summary.append(f"- {name} = {value}")
        if len(summary) == 1:
            summary.append("- None")
        return summary
    
    summary = []
    
    global_vars = [(k, v) for k, v in globals().items() if not k.startswith('__') and k != "AgentBase"]
    functions = [(k, v) for k, v in global_vars if inspect.isfunction(v)]
    calsses = [(k, v) for k, v in global_vars if inspect.isclass(v)]
    modules = [(k, v) for k, v in global_vars if inspect.ismodule(v)]
    variables = [(k, v) for k, v in global_vars if not (inspect.isfunction(v) or inspect.isclass(v) or inspect.ismodule(v))]
    
    summary.extend(summarize_items(functions, "\nGlobal Functions:"))
    summary.extend(summarize_items(modules, "\nGlobal Modules:"))
    summary.extend(summarize_items(variables, "\nGlobal Variables:"))
    summary.extend(summarize_items(calsses, "\nGlobal Classes:"))
    
    methods = inspect.getmembers(agent, inspect.ismethod)
    attributes = inspect.getmembers(agent, lambda x: not inspect.ismethod(x))
    
    summary.extend(summarize_items(methods, "\nCurrent Agent Instance's Methods:"))
    summary.extend(summarize_items(attributes, "\nCurrent Agent Instance's Attributes:"))

    return "\n".join(summary).strip()

def action_read_logic(module_name: str, target_name: str):
    """
    Reads the source code of the specified logic (function, method, or class) within a given module.
    
    Args:
        module_name (str): The name of the module (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class (e.g., 'solver', 'Agent.action_call_llm', 'Agent').
    
    Returns:
        code_str (str): A string representing the source code of the specified logic.
    """
    # Dynamically import the module
    module = importlib.import_module(module_name)
    
    # If target_name contains a dot, it's a method in a class (e.g., 'Agent.evolve')
    if '.' in target_name:
        class_name, target_name = target_name.split('.')
        target_class = getattr(module, class_name)
        target = getattr(target_class, target_name)
    else:
        # Otherwise, it's a top-level function or class
        target = getattr(module, target_name)
    
    # Extract the source code using inspect
    code_str = logic.get_source_code(target, target_name)
    
    return code_str
    
def action_adjust_logic(module_name: str, target_name: str, new_code=str, target_type: str = 'function', operation: str = 'modify'):
    """
    Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to 
    improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.

    Args:
        module_name (str): The name of the module to modify (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class to do operation (e.g., 'solver').
        new_code (str): The new logic as a string (including `def` for functions or `class` or classes). For delete, it can be empty string.
        target_type (str): The type of target ('function', 'class'). Default is 'function'.
        operation (str): The type of operation to perform ('modify', 'add', or 'delete'). Default is 'modify'.

    Raises:
        ValueError: Unknown operation

    Examples:
        >>> modify_logic('agent_module', 'evolve', 'def evolve(agent):\\n    print("New evolve method")', target_type='function')
        >>> modify_logic('agent_module', 'evolve', '', target_type='function', operation='delete')
    """
    if module_name == "agent_module":
        if target_name == "solver":
            if "time.sleep" in new_code:
                raise ValueError("Don't use `time.sleep` in solver.")
        if target_name == "Agent.action_call_llm":
            raise ValueError("Don't modify `action_call_llm`.")
        if target_name == "Agent.action_call_json_format_llm":
            raise ValueError("Don't modify `action_call_json_format_llm`.")

    if "import logging" in new_code or "from logging" in new_code:
        raise ValueError("Don't use `logging`.")

    # Import the module dynamically
    module = importlib.import_module(module_name)
    _target_name = target_name
    print(f"========== New Code:  ==========\n{new_code}", end='\n\n')
    # Perform the operation based on type (modify, add, delete)
    if operation in ['modify', 'add']:
        # Compile the new code within the current global and a new local dict
        locals_dict = {}
        try:
            compiled = compile(new_code, f"running.{module_name}.{target_name}", "exec")
        except SyntaxError as e:
            line = (e.text or "").rstrip()
            location = f"line {e.lineno}"
            if e.offset is not None:
                location += f", column {e.offset}"
            detail = f"Invalid Python for `{module_name}.{_target_name}` at {location}: {e.msg}"
            if line:
                detail += f"\n{line}"
            raise ValueError(detail) from e
        exec(compiled, globals(), locals_dict)
        if '.' in target_name:
            class_name_, target_name_ = target_name.split('.')
            if class_name_ in locals_dict:
                new_target = getattr(locals_dict[class_name_], target_name_)
                locals_dict.pop(class_name_)
            else:
                new_target = locals_dict[target_name_]
                locals_dict.pop(target_name_)
        else:
            new_target = locals_dict[target_name]
            locals_dict.pop(target_name)
        globals().update(locals_dict)
        # Apply the new definition or value to the target
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            setattr(cls, target_name, new_target)
            getattr(cls, target_name).__source__ = new_code
            # Add or update the __source__ attribute on the class level to store the full new definition
            if not hasattr(cls, '__source__'):
                cls.__source__ = {}
                for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    cls.__source__[name] = logic.get_source_code(method, name)
            cls.__source__[target_name] = '\n'.join(['    ' + code_line for code_line in new_code.split('\n')])

        else:  # Module level attribute
            setattr(module, target_name, new_target)
            getattr(module, target_name).__source__ = new_code

    elif operation == 'delete':
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            delattr(cls, target_name)
            if hasattr(cls, '__source__') and target_name in cls.__source__:
                del cls.__source__[target_name]
        else:  # Module level attribute
            delattr(module, target_name)

    else:
        raise ValueError(f"Unknown operation '{operation}'. Expected 'modify', 'add', or 'delete'.")

    return f"Successfully {operation} `{module_name}.{_target_name}`."

def action_run_code(code_type: str, code: str, timeout: float = 30.0, agent=None) -> str:
    """
    Execute Python or shell code and capture the output, errors, and return value. 
    (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)
    
    Args:
        code_type (str): The type of code to execute ('python' or 'bash').
        code (str): The code to execute as a string.
        timeout (float): Maximum execution time in seconds (default: 30.0).
    
    Returns:
        result_str (str): A string summarizing the output, errors, and return value.
    """
    
    def safe_eval(expr: str, globals_dict, locals_dict):
        """Safely evaluate an expression."""
        try:
            tree = ast.parse(expr, mode='eval')
            return eval(compile(tree, '<string>', 'eval'), globals_dict, locals_dict)
        except Exception:
            return None

    if code_type.lower() == 'python':
        output = io.StringIO()
        error_output = io.StringIO()
        return_value = None
        locals_dict = {
            "agent": agent,
            "self": agent,
            "self_evolving_agent": agent,
        }
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
                # Provide the active agent under the names that the evolving model
                # is most likely to reference in ad-hoc Python snippets.
                exec(code, globals(), locals_dict)
                globals().update(
                    {
                        key: value
                        for key, value in locals_dict.items()
                        if key not in {"agent", "self", "self_evolving_agent"}
                    }
                )

                lines = code.splitlines()
                if lines:
                    # Safely evaluate the last expression
                    return_value = safe_eval(lines[-1], globals(), locals_dict)
            result = {
                "output": output.getvalue(),
                "errors": error_output.getvalue(),
                "return_value": return_value
            }
        except Exception:
            result = {
                "output": output.getvalue(),
                "errors": error_output.getvalue() + _format_exception("Python execution error:\n"),
                "return_value": None,
            }
        finally:
            output.close()
            error_output.close()
    
    elif code_type.lower() == 'bash':
        try:
            process = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            result = {
                "output": process.stdout,
                "errors": process.stderr,
                "return_value": process.returncode
            }
        except subprocess.TimeoutExpired:
            result = {
                "output": "",
                "errors": f"Command timed out after {timeout} seconds",
                "return_value": None
            }
        except Exception as e:
            result = {
                "output": "",
                "errors": repr(e),
                "return_value": None
            }
            
    else:
        return "Error: Unsupported code_type. Only 'python' and 'bash' are supported."

    # Format the result
    result_str = f"Execution Summary ({code_type.capitalize()}):\n"
    if result["output"]:
        result_str += f"Output:\n{result['output']}\n"
    if result["errors"]:
        result_str += f"Errors:\n{result['errors']}\n"
    if result["return_value"] is not None:
        result_str += f"Return Value: {result['return_value']}\n"
    
    return result_str or "No output, errors, or return value."

def solver(agent, task: str):
    prompt_cfg = agent.task_module.normalize_prompt_config(getattr(agent, "prompt_config", {}))

    # IFEval: use raw text mode to avoid JSON wrapping conflicting with constraints
    if getattr(agent, "task_name", None) == "ifeval":
        system_prompt = (
            f"You are a helpful {prompt_cfg['role']}.\n"
            f"Requirements:\n{prompt_cfg['requirements']}"
        ).strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        responses = agent.action_call_llm(
            model=agent.default_infer_model,
            messages=messages,
            temperature=prompt_cfg["temperature"],
            n=1,
            response_format="text",
        )
        raw = responses[0]
        response_text = str(getattr(raw, "content", raw) if not isinstance(raw, str) else raw).strip()
        return {"response": response_text, "answer": response_text}

    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model=agent.default_infer_model,
        messages=messages, 
        temperature=prompt_cfg["temperature"], 
        num_of_response=1,
        role=prompt_cfg["role"], 
        return_dict_keys=["reasoning", "answer", "response"], 
        requirements=prompt_cfg["requirements"],
    )

    return_dict = response[0] if response else {}
    output_coercer = getattr(agent.task_module, "_coerce_solver_output", None)
    if getattr(agent, "task_name", None) == "mmlu" and callable(output_coercer):
        return output_coercer(task, return_dict)

    if not isinstance(return_dict, dict):
        return_dict = {"answer": str(return_dict or "")}

    final_text = str(
        return_dict.get("response", return_dict.get("answer", ""))
    ).strip()
    return_dict["answer"] = final_text
    return_dict["response"] = final_text
    return return_dict


def action_evaluate_on_task(agent: AgentBase, task, solver):
    """
    Evaluate the current solver on the goal task samples and return evaluation feedback.
    """
    global best_eval_acc
    os.makedirs("results", exist_ok=True)
    selected_eval_examples = agent.selected_examples.get("evaluate", {}).get("examples")
    feedback, acc = task.evaluate(solver, examples_override=selected_eval_examples,
                                   max_workers=getattr(agent, "max_workers", 48))
    prev_eval_acc = agent.credit_state.get("last_eval_acc")
    agent.credit_state["last_eval_acc"] = float(acc)

    if agent.credit_state.get("pending_structure_eval", False):
        baseline = agent.credit_state.get("structure_eval_baseline")
        if baseline is None:
            baseline = prev_eval_acc
        if baseline is not None:
            agent.credit_state["last_structure_delta"] = float(acc) - float(baseline)
        agent.credit_state["pending_structure_eval"] = False

    if acc > best_eval_acc:
        if best_eval_acc <= 0.0 or acc >= best_eval_acc + logic_dump_improvement_epsilon:
            logic.store_all_logic(f"./{agent.task_name}_{round(acc, 4)}")
        best_eval_acc = acc

    credit_report = action_get_evolution_credit(agent)
    reflection_summary = _summarize_evaluation_feedback(agent, feedback)
    agent.credit_state["last_failure_reflection"] = reflection_summary

    # Inline the failure taxonomy + recommendations that action_display_analysis
    # would produce, so the agent never needs a separate tool call for it.
    analysis_report = action_display_analysis(agent, analysis="(auto-generated by action_evaluate_on_task)")

    return (
        f"{feedback}\n\n"
        f"{reflection_summary}\n\n"
        f"{analysis_report}\n\n"
        f"Evolution Credit Report:\n{credit_report}"
    )


def action_optimize_prompt_on_task(
    agent: AgentBase,
    iterations: int = None,
    train_size: int = None,
    valid_size: int = None,
    candidates_per_iter: int = None,
    seed: int = None,
):
    """
    Run inner-loop prompt optimization for the current task and update agent.prompt_config.
    """
    previous_prompt_config = dict(agent.prompt_config)
    defaults = agent.inner_loop_defaults
    iterations = defaults["iterations"] if iterations is None else iterations
    train_size = defaults["train_size"] if train_size is None else train_size
    valid_size = defaults["valid_size"] if valid_size is None else valid_size
    candidates_per_iter = defaults["candidates_per_iter"] if candidates_per_iter is None else candidates_per_iter
    base_seed = int(defaults.get("seed", 7)) if seed is None else int(seed)
    rotate_seed_each_call = bool(defaults.get("rotate_seed_each_call", True))
    optimization_seed = base_seed
    if rotate_seed_each_call:
        optimization_seed += int(agent.credit_state.get("inner_steps", 0))
    selected_train_examples = agent.selected_examples.get("train", {}).get("examples")
    selected_valid_examples = agent.selected_examples.get("valid", {}).get("examples")

    report, report_text = agent.task_module.optimize_prompt_config(
        agent=agent,
        current_prompt_config=agent.prompt_config,
        iterations=iterations,
        train_size=train_size,
        valid_size=valid_size,
        candidates_per_iter=candidates_per_iter,
        seed=optimization_seed,
        selected_train_examples=selected_train_examples,
        selected_valid_examples=selected_valid_examples,
    )
    initial_valid_accuracy = float(report.get("initial_valid_accuracy", 0.0))
    best_valid_accuracy = float(report.get("best_valid_accuracy", 0.0))
    delta_u_p = float(report.get("delta_u_p", 0.0))
    if best_valid_accuracy >= initial_valid_accuracy:
        agent.prompt_config = report["best_prompt_config"]
    else:
        agent.prompt_config = previous_prompt_config
    agent.credit_state["last_prompt_delta"] = delta_u_p
    agent.credit_state["last_prompt_before"] = initial_valid_accuracy
    agent.credit_state["last_prompt_after"] = best_valid_accuracy
    agent.credit_state["prompt_saturated"] = 0.0 <= delta_u_p < float(agent.prompt_saturation_epsilon)
    agent.credit_state["last_prompt_report"] = report
    agent.credit_state["inner_steps"] = int(agent.credit_state.get("inner_steps", 0)) + 1

    saturation_text = (
        "yes" if agent.credit_state["prompt_saturated"] else "no"
    )
    regression_text = "yes" if best_valid_accuracy < initial_valid_accuracy else "no"
    return (
        f"{report_text}\n"
        f"Selection seed: {optimization_seed}\n"
        f"Delta_U_P={delta_u_p:.4f}, epsilon={agent.prompt_saturation_epsilon:.4f}, "
        f"prompt_saturated={saturation_text}, prompt_regressed={regression_text}"
    )


def action_get_evolution_credit(agent: AgentBase):
    epsilon = float(agent.prompt_saturation_epsilon)
    delta_u_p = float(agent.credit_state.get("last_prompt_delta", 0.0))
    delta_u_c = agent.credit_state.get("last_structure_delta")
    if delta_u_c is None:
        delta_u_c_text = "N/A"
    else:
        delta_u_c_text = f"{float(delta_u_c):.4f}"
    return (
        f"Delta_U_P={delta_u_p:.4f}\n"
        f"Delta_U_C={delta_u_c_text}\n"
        f"Epsilon={epsilon:.4f}\n"
        f"PromptSaturated={agent.credit_state.get('prompt_saturated', False)}\n"
        f"PendingStructureEval={agent.credit_state.get('pending_structure_eval', False)}\n"
        f"LastEvalAcc={agent.credit_state.get('last_eval_acc')}"
    )


def _load_example_pool(agent: AgentBase, split: str):
    split = str(split or "train").lower().strip()
    if agent.task_name == "mmlu":
        examples = list(agent.task_module._load_mmlu_examples())
        if split in {"train", "valid", "in_domain"}:
            return examples[:128]
        if split in {"test", "evaluate"}:
            return examples[128:928]
    elif agent.task_name == "mgsm":
        examples = list(agent.task_module.get_all_examples())
        random.seed(0)
        random.shuffle(examples)
        if split in {"train", "valid", "in_domain"}:
            return examples[:128]
        if split in {"test", "evaluate"}:
            return examples[128:928]
    elif agent.task_name == "ifeval":
        if split in {"train", "valid", "in_domain"}:
            return list(agent.task_module._load_ifeval_examples("train"))
        if split in {"test", "evaluate"}:
            return list(agent.task_module._load_ifeval_examples("test"))
    raise ValueError(f"Unsupported split `{split}` for task `{agent.task_name}`.")


def _example_group_key(agent: AgentBase, example: typing.Dict[str, typing.Any]) -> str:
    if agent.task_name == "mmlu":
        return str(example.get("Subject", "unknown"))
    if agent.task_name == "mgsm":
        return str(example.get("lang", "unknown"))
    if agent.task_name == "ifeval":
        instruction_ids = example.get("instruction_id_list", [])
        if instruction_ids:
            return str(instruction_ids[0])
        return str(example.get("key", "unknown"))
    return "default"


def _example_preview(agent: AgentBase, example: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    if agent.task_name == "mmlu":
        return {
            "subject": example.get("Subject"),
            "question": str(example.get("Question", ""))[:160],
            "answer": example.get("Answer"),
        }
    if agent.task_name == "mgsm":
        return {
            "lang": example.get("lang"),
            "input": str(example.get("inputs", ""))[:160],
            "target": example.get("targets"),
        }
    if agent.task_name == "ifeval":
        return {
            "key": example.get("key"),
            "instruction_id_list": list(example.get("instruction_id_list", []))[:4],
            "prompt": str(example.get("prompt", ""))[:160],
        }
    return {"preview": str(example)[:160]}


def _select_examples_from_pool(
    agent: AgentBase,
    examples: typing.List[typing.Dict[str, typing.Any]],
    count: int,
    strategy: str,
    seed: int,
):
    count = max(0, int(count))
    strategy = str(strategy or "random").lower().strip()
    rng = random.Random(int(seed))
    pool = list(examples)
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return pool

    if strategy == "head":
        return pool[:count]
    if strategy == "random":
        rng.shuffle(pool)
        return pool[:count]
    if strategy == "diverse":
        grouped = collections.defaultdict(list)
        for example in pool:
            grouped[_example_group_key(agent, example)].append(example)
        group_keys = list(grouped.keys())
        rng.shuffle(group_keys)
        for key in group_keys:
            rng.shuffle(grouped[key])
        selected = []
        while len(selected) < count:
            progressed = False
            for key in group_keys:
                if grouped[key]:
                    selected.append(grouped[key].pop())
                    progressed = True
                    if len(selected) >= count:
                        break
            if not progressed:
                break
        return selected
    raise ValueError(f"Unknown selection strategy `{strategy}`. Use head, random, or diverse.")


def action_select_examples(
    agent: AgentBase,
    split: str = "train",
    count: int = 32,
    strategy: str = "diverse",
    persist_for: str = None,
    seed: int = 0,
):
    """
    Select a task-aware subset of examples and optionally persist it for future
    train/valid/evaluate calls.
    """
    split = str(split or "train").lower().strip()
    strategy = str(strategy or "diverse").lower().strip()
    if persist_for is None:
        persist_for = "evaluate" if split in {"test", "evaluate"} else split
    persist_for = str(persist_for).lower().strip()
    if persist_for == "in_domain":
        persist_for = "train"
    if persist_for == "optimize":
        if split == "valid":
            persist_for = "valid"
        elif split in {"test", "evaluate"}:
            persist_for = "evaluate"
        else:
            persist_for = "train"
    if persist_for not in {"train", "valid", "evaluate", "none"}:
        raise ValueError("persist_for must be one of: train, valid, evaluate, optimize, none.")

    if int(count) <= 0:
        if persist_for != "none":
            agent.selected_examples.pop(persist_for, None)
            return f"Cleared stored example selection for `{persist_for}`."
        return "No examples selected because count <= 0."

    pool = _load_example_pool(agent, split)
    selected = _select_examples_from_pool(
        agent=agent,
        examples=pool,
        count=count,
        strategy=strategy,
        seed=seed,
    )
    if persist_for != "none":
        agent.selected_examples[persist_for] = {
            "split": split,
            "strategy": strategy,
            "seed": int(seed),
            "examples": selected,
        }

    group_counts = collections.Counter(_example_group_key(agent, example) for example in selected)
    preview = [_example_preview(agent, example) for example in selected[:3]]
    storage_note = (
        f"Stored for future `{persist_for}` calls."
        if persist_for != "none"
        else "Preview only; not persisted."
    )
    return (
        f"Selected {len(selected)} examples from `{split}` using `{strategy}` strategy.\n"
        f"{storage_note}\n"
        f"Top groups: {json.dumps(dict(group_counts.most_common(5)), ensure_ascii=True)}\n"
        f"Preview: {json.dumps(preview, ensure_ascii=True)}"
    )


# ---------------------------------------------------------------------------
# action_compare_variants – A/B evaluation of two solver or prompt variants
# ---------------------------------------------------------------------------

def _build_variant_solver(agent: AgentBase, spec: typing.Dict[str, typing.Any]):
    """
    Build a callable solver from a variant spec.

    Supported spec keys (all optional – omitted keys fall back to current agent state):
        prompt_config  – dict with role / requirements / temperature overrides
        solver_code    – full Python source of a replacement solver function
    """
    prompt_override = spec.get("prompt_config")
    solver_code = spec.get("solver_code")

    if solver_code:
        # Compile a one-off solver from source
        local_ns: typing.Dict[str, typing.Any] = {}
        compiled = compile(solver_code, "<variant_solver>", "exec")
        exec(compiled, globals(), local_ns)
        custom_fn = local_ns.get("solver")
        if custom_fn is None:
            raise ValueError(
                "solver_code must define a top-level `solver(agent, task)` function."
            )
        return functools.partial(custom_fn, agent)

    if prompt_override:
        merged = dict(agent.prompt_config)
        merged.update({k: v for k, v in prompt_override.items() if v is not None})

        def _prompt_variant_solver(task: str):
            cfg = agent.task_module.normalize_prompt_config(merged)
            messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
            response = agent.action_call_json_format_llm(
                model=agent.default_infer_model,
                messages=messages,
                temperature=cfg["temperature"],
                num_of_response=1,
                role=cfg["role"],
                return_dict_keys=["reasoning", "answer", "response"],
                requirements=cfg["requirements"],
            )
            rd = response[0]
            final = str(rd.get("response", rd.get("answer", ""))).strip()
            rd["answer"] = final
            rd["response"] = final
            return rd

        return _prompt_variant_solver

    # No overrides – just use the current solver
    return functools.partial(solver, agent)


def _fast_solver_eval(
    agent: AgentBase,
    solver_fn,
    examples: typing.List[typing.Dict[str, typing.Any]],
):
    task_name = getattr(agent, "task_name", "")
    task_module = agent.task_module
    correctness = []
    info_list = []

    for idx, example in enumerate(examples):
        try:
            if task_name == "mmlu":
                prompt = task_module.format_multichoice_question(example)
                res = solver_fn(prompt)
                predicted = task_module._extract_answer(
                    res["answer"],
                    task_text=prompt,
                )
                gold = str(example["Answer"])
                is_correct = predicted == gold
                info_list.append(
                    f"Valid Sample {idx}:\n{prompt}\nModel Output: {res}\n"
                    f"Model Answer: {predicted}\nCorrect Answer: {gold}\nIs Correct: {is_correct}\n"
                )
            elif task_name == "mgsm":
                prompt = example["inputs"]
                res = solver_fn(prompt)
                predicted = task_module._extract_numeric_answer(res["answer"])
                gold = str(example["targets"])
                is_correct = task_module.score_mgsm(gold, predicted)
                info_list.append(
                    f"Valid Sample {idx}:\n{prompt}\nModel Output: {res}\n"
                    f"Model Answer: {predicted}\nCorrect Answer: {gold}\nIs Correct: {is_correct}\n"
                )
            elif task_name == "ifeval":
                prompt = example["prompt"]
                res = solver_fn(prompt)
                response_text = str(res.get("response", res) if isinstance(res, dict) else res)
                strict, loose, per_instr = task_module.verify_all_instructions(
                    response_text,
                    example["instruction_id_list"],
                    example["kwargs"],
                )
                is_correct = bool(strict)
                failed_ids = [
                    iid for iid, passed in zip(example["instruction_id_list"], per_instr) if not passed
                ]
                info_list.append(
                    f"Valid Sample {idx} (key={example['key']}):\n"
                    f"Prompt: {prompt[:200]}...\n"
                    f"Response: {response_text[:300]}...\n"
                    f"Strict: {strict}  Loose: {loose:.2f}  Failed: {failed_ids}\n"
                )
            else:
                raise ValueError(f"Fast comparison is not implemented for task `{task_name}`.")
        except Exception as e:
            is_correct = False
            info_list.append(f"Valid Sample {idx}:\n{repr(e)}\n")
        correctness.append(bool(is_correct))

    accuracy = float(sum(correctness) / len(correctness)) if correctness else 0.0
    return {
        "accuracy": accuracy,
        "correctness": correctness,
        "feedback": "\n".join(info_list),
    }


def _format_compare_report(metrics: typing.Dict[str, typing.Any]) -> str:
    label_a = metrics["label_a"]
    label_b = metrics["label_b"]
    delta = metrics["delta"]
    if delta > 0.02:
        recommendation = f"Variant {label_b} is clearly better (+{delta:.4f}). Adopt it."
    elif delta < -0.02:
        recommendation = f"Variant {label_a} is clearly better ({delta:.4f} for B). Keep current."
    else:
        recommendation = (
            f"Variants are within noise (Δ={delta:+.4f}). "
            "Prefer the simpler or lower-risk option, or increase sample_count for a clearer signal."
        )

    sections = [
        "=== Variant Comparison ===",
        f"Samples: {metrics['sample_count']}  (seed={metrics['seed']})",
        "",
        f"  [{label_a}] accuracy={metrics['acc_a']:.4f}",
        f"  [{label_b}] accuracy={metrics['acc_b']:.4f}",
        f"  Δ (B−A) = {delta:+.4f}",
        "",
        "=== Per-Sample Diff ===",
        f"  Both correct:  {metrics['both_right']}",
        f"  {label_a} only correct: {metrics['a_only']}",
        f"  {label_b} only correct: {metrics['b_only']}",
        f"  Both wrong:    {metrics['both_wrong']}",
        "",
        "=== Recommendation ===",
        recommendation,
    ]
    return "\n".join(sections)


def _compare_variant_metrics(
    agent: AgentBase,
    variant_a: typing.Dict[str, typing.Any],
    variant_b: typing.Dict[str, typing.Any],
    sample_count: int = 20,
    seed: int = 42,
):
    label_a = str(variant_a.get("label", "A"))
    label_b = str(variant_b.get("label", "B"))
    sample_count = max(1, int(sample_count))

    pool = _load_example_pool(agent, "train")
    shared = _select_examples_from_pool(
        agent=agent,
        examples=pool,
        count=sample_count,
        strategy="diverse",
        seed=int(seed),
    )
    if not shared:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "sample_count": 0,
            "seed": int(seed),
            "acc_a": 0.0,
            "acc_b": 0.0,
            "delta": 0.0,
            "both_right": 0,
            "a_only": 0,
            "b_only": 0,
            "both_wrong": 0,
            "feedback_a": "No examples available for comparison.",
            "feedback_b": "No examples available for comparison.",
        }

    solver_a = _build_variant_solver(agent, variant_a)
    solver_b = _build_variant_solver(agent, variant_b)
    eval_a = _fast_solver_eval(agent, solver_a, shared)
    eval_b = _fast_solver_eval(agent, solver_b, shared)
    correct_a = eval_a["correctness"]
    correct_b = eval_b["correctness"]
    n = min(len(correct_a), len(correct_b), len(shared))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "sample_count": n,
        "seed": int(seed),
        "acc_a": float(eval_a["accuracy"]),
        "acc_b": float(eval_b["accuracy"]),
        "delta": float(eval_b["accuracy"] - eval_a["accuracy"]),
        "both_right": sum(1 for i in range(n) if correct_a[i] and correct_b[i]),
        "a_only": sum(1 for i in range(n) if correct_a[i] and not correct_b[i]),
        "b_only": sum(1 for i in range(n) if correct_b[i] and not correct_a[i]),
        "both_wrong": sum(1 for i in range(n) if not correct_a[i] and not correct_b[i]),
        "feedback_a": eval_a["feedback"],
        "feedback_b": eval_b["feedback"],
    }


def action_compare_variants(
    agent: AgentBase,
    variant_a: typing.Dict[str, typing.Any],
    variant_b: typing.Dict[str, typing.Any],
    sample_count: int = 20,
    seed: int = 42,
) -> str:
    """
    Run an A/B comparison of two solver or prompt variants on the same sampled
    validation subset so the agent can make an informed evolve-or-keep decision.

    Each variant dict may contain:
        label         – human-readable name (default "A" / "B")
        prompt_config – dict overriding role / requirements / temperature
        solver_code   – full Python source of a replacement solver(agent, task)

    Omitted keys fall back to the agent's current state, so passing an empty
    dict for variant_a effectively means "current solver".

    Returns a structured comparison report with per-variant accuracy, a
    per-sample diff, and a recommendation.
    """
    metrics = _compare_variant_metrics(
        agent=agent,
        variant_a=variant_a,
        variant_b=variant_b,
        sample_count=sample_count,
        seed=seed,
    )
    return _format_compare_report(metrics)


def _summarize_evaluation_feedback(agent: AgentBase, feedback: str) -> str:
    """
    Build a compact reflection summary from task evaluation feedback so the next
    outer-loop step receives an explicit evolution direction instead of raw logs only.
    """
    text = str(feedback or "")
    if not text.strip():
        return "Reflection Summary:\nNo evaluation feedback available."

    metric_patterns = [
        (r"Valid Strict Accuracy:\s*([0-9.]+)", "valid_strict_accuracy"),
        (r"Valid Loose Accuracy:\s*([0-9.]+)", "valid_loose_accuracy"),
        (r"Test Strict Accuracy:\s*([0-9.]+)", "test_strict_accuracy"),
        (r"Cached Test Strict Accuracy:\s*([0-9.]+)", "cached_test_strict_accuracy"),
        (r"Valid Accuracy:\s*([0-9.]+)", "valid_accuracy"),
        (r"Test Accuracy\s*([0-9.]+)", "test_accuracy"),
    ]
    metrics = {}
    for pattern, key in metric_patterns:
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))

    case_blocks = re.split(r"(?=(?:Valid Sample|Sample)\s+\d+(?:\s*\([^)]*\))?:)", text)
    case_blocks = [
        block.strip()
        for block in case_blocks
        if re.match(r"^(?:Valid Sample|Sample)\s+\d+(?:\s*\([^)]*\))?:", block.strip())
    ]

    total_cases = len(case_blocks)
    incorrect_cases = 0
    exception_cases = 0
    explicit_fail_cases = 0
    extraction_cases = 0
    format_cases = 0
    reasoning_cases = 0
    failed_instruction_ids = collections.Counter()

    extraction_markers = [
        "NO response IN DICTIONARY",
        "KeyError('answer')",
        'KeyError("answer")',
        "KeyError('response')",
        'KeyError("response")',
        "Model Output: None",
        "Model Output: {}",
    ]
    format_markers = [
        "<<",
        ">>>",
        "***",
        "markdown",
        "repeat EXACTLY",
        "word-for-word",
        "separator",
        "keyword",
        "format",
        "json",
    ]

    for block in case_blocks:
        has_exception = "Traceback" in block or "Exception" in block or "Error" in block
        if has_exception:
            exception_cases += 1

        is_incorrect = False
        match = re.search(r"Is Correct:\s*(True|False)", block)
        if match:
            is_incorrect = match.group(1) == "False"
        elif re.search(r"Strict:\s*0\b", block):
            is_incorrect = True

        failed_match = re.search(r"Failed:\s*\[([^\]]*)\]", block)
        if failed_match:
            ids = [item.strip().strip("'\"") for item in failed_match.group(1).split(",") if item.strip()]
            if ids:
                explicit_fail_cases += 1
                for item in ids:
                    failed_instruction_ids[item] += 1

        lowered = block.lower()
        has_extraction_issue = any(marker.lower() in lowered for marker in extraction_markers)
        if not has_extraction_issue and "model output:" in lowered and (
            "'answer': ''" in lowered
            or '"answer": ""' in lowered
            or "'response': ''" in lowered
            or '"response": ""' in lowered
        ):
            has_extraction_issue = True

        has_format_issue = any(marker.lower() in lowered for marker in format_markers) or failed_match is not None

        if is_incorrect:
            incorrect_cases += 1
            if has_extraction_issue or has_exception:
                extraction_cases += 1
            elif has_format_issue:
                format_cases += 1
            else:
                reasoning_cases += 1

    dominant_modes = []
    if extraction_cases:
        dominant_modes.append(f"extraction/runtime={extraction_cases}")
    if format_cases:
        dominant_modes.append(f"format/constraint={format_cases}")
    if reasoning_cases:
        dominant_modes.append(f"reasoning/content={reasoning_cases}")
    if not dominant_modes and incorrect_cases:
        dominant_modes.append(f"unclassified={incorrect_cases}")
    if not dominant_modes:
        dominant_modes.append("no explicit failures captured")

    top_instruction_ids = ", ".join(
        f"{name} x{count}" for name, count in failed_instruction_ids.most_common(4)
    )
    if not top_instruction_ids:
        top_instruction_ids = "none"

    current_eval = (
        metrics.get("valid_strict_accuracy")
        if "valid_strict_accuracy" in metrics
        else metrics.get("valid_accuracy")
    )
    prompt_saturated = bool(agent.credit_state.get("prompt_saturated", False))

    next_steps = []
    if extraction_cases > 0:
        next_steps.append(
            "Harden solver output handling and prompt requirements so the final answer is always emitted in the expected key/format."
        )
    if format_cases > 0:
        next_steps.append(
            "Prioritize prompt-level fixes that restate formatting and constraint-following requirements more explicitly."
        )
    if reasoning_cases > 0:
        next_steps.append(
            "Consider reasoning-oriented changes such as decomposition, verification, or better answer selection logic."
        )
    if current_eval is not None and current_eval >= 0.9:
        next_steps.append("Performance is already strong; prefer conservative, low-risk changes and verify regressions carefully.")
    elif prompt_saturated:
        next_steps.append(
            "Prompt gains appear saturated, so structural edits or solver-side checks are now better candidates than more prompt-only tuning."
        )
    else:
        next_steps.append(
            "Prompt gains are not saturated yet, so favor inner-loop prompt optimization before risky structural edits."
        )

    return (
        "Reflection Summary:\n"
        f"- Metrics: {json.dumps(metrics, sort_keys=True) if metrics else 'none parsed'}\n"
        f"- Cases reviewed: {total_cases}, incorrect: {incorrect_cases}, exceptions: {exception_cases}, explicit_failed_cases: {explicit_fail_cases}\n"
        f"- Dominant failure modes: {', '.join(dominant_modes)}\n"
        f"- Frequent failed instruction ids: {top_instruction_ids}\n"
        f"- Evolution direction: {' '.join(next_steps)}"
    )


def _repair_json_escapes(text: str) -> str:
    """
    Fix invalid JSON escape sequences produced by some models (e.g. Mistral).
    Replaces lone backslashes that don't form valid JSON escapes with double
    backslashes so json.loads can parse the string.
    Also escapes raw control characters (e.g. literal newlines/tabs inside
    JSON string values) that cause 'Invalid control character' errors.
    """
    # First pass: escape raw control characters (U+0000–U+001F) that are not
    # already part of a valid JSON escape sequence.
    _CTRL_MAP = {
        '\n': '\\n', '\r': '\\r', '\t': '\\t',
        '\b': '\\b', '\f': '\\f',
    }
    sanitized = []
    for ch in text:
        if ch in _CTRL_MAP:
            sanitized.append(_CTRL_MAP[ch])
        elif ord(ch) < 0x20:
            sanitized.append(f'\\u{ord(ch):04x}')
        else:
            sanitized.append(ch)
    text = ''.join(sanitized)

    # Second pass: fix invalid backslash escapes.
    _VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in _VALID_JSON_ESCAPES:
                out.append(ch)
                out.append(nxt)
                i += 2
            else:
                # Invalid escape like \S, \p, etc. — double the backslash
                out.append('\\\\')
                i += 1
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def _extract_first_json_object(text: str):
    """
    Best-effort extraction for the first complete JSON object in free-form text.
    Handles fenced blocks and assistant chatter around JSON.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
    if fenced:
        s = fenced.group(1).strip()

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None

class Agent(AgentBase):
    def __init__(
        agent,
        api_key=None,
        goal_prompt_path='src/goal_prompt.md',
        key_path=None,
        task_name: str = "mmlu",
        max_outer_evolve_steps: int = 10,
        inner_loop_defaults: typing.Optional[typing.Dict[str, int]] = None,
        prompt_saturation_epsilon: float = 0.01,
        fallback_min_prompt_update_rounds: int = 2,
        ab_gate_delta_threshold: float = 0.02,
        max_input_tokens: int = 128000,
        max_output_tokens: int = 4000,
        max_workers: int = 48,
    ):
        # Load configurations
        with open(goal_prompt_path, encoding='utf-8') as prompt_file:
            agent.goal_prompt = prompt_file.read()
        task_name = str(task_name).lower().strip()
        if task_name not in TASK_MODULES:
            raise ValueError(f"Unknown task_name={task_name}. Supported tasks: {sorted(TASK_MODULES.keys())}")
        agent.task_name = task_name
        agent.task_module = TASK_MODULES[task_name]
        task_class_name = f"{task_name.upper()}_Task"
        agent.goal_task = getattr(agent.task_module, task_class_name)()
        agent.prompt_config = dict(agent.task_module.DEFAULT_PROMPT_CONFIG)
        agent.selected_examples = {}
        agent.max_outer_evolve_steps = int(max_outer_evolve_steps)
        defaults = {
            "iterations": 4,
            "train_size": 32,
            "valid_size": 32,
            "candidates_per_iter": 8,
            "seed": 7,
            "rotate_seed_each_call": True,
        }
        if isinstance(inner_loop_defaults, dict):
            defaults.update({k: inner_loop_defaults[k] for k in defaults.keys() if k in inner_loop_defaults})
        agent.inner_loop_defaults = defaults
        agent.prompt_saturation_epsilon = float(prompt_saturation_epsilon)
        agent.fallback_min_prompt_update_rounds = max(1, int(fallback_min_prompt_update_rounds))
        agent.ab_gate_delta_threshold = float(ab_gate_delta_threshold)
        agent.max_input_tokens = max(1024, int(max_input_tokens))
        agent.max_output_tokens = max(256, int(max_output_tokens))
        agent.max_workers = max(1, int(max_workers))
        default_model = "qwen-local"
        agent.default_infer_model = os.getenv("AGENT_MODEL_NAME", default_model)
        agent.optimization_model = os.getenv("OPTIMIZER_MODEL_NAME", agent.default_infer_model)
        agent.outer_loop_model = os.getenv("OUTER_LOOP_MODEL_NAME", agent.optimization_model)
        if openai is None:
            raise RuntimeError(
                "vLLM mode requires `openai` package for OpenAI-compatible API calls. "
                "Install with `pip install openai`."
            )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None and key_path and os.path.exists(key_path):
                with open(key_path, encoding='utf-8') as key_file:
                    api_key = key_file.read().strip()
        if not api_key:
            api_key = "EMPTY"
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1"
        openai.api_key = api_key

        # Build a list of clients for round-robin load balancing.
        base_urls_env = os.getenv("OPENAI_BASE_URLS", "")
        base_urls = [u.strip() for u in base_urls_env.split(",") if u.strip()] if base_urls_env else []
        if not base_urls:
            base_urls = [base_url] if base_url else []

        agent._clients = [openai.OpenAI(api_key=api_key, base_url=u, timeout=30.0) for u in base_urls] \
            if base_urls else [openai.OpenAI(api_key=api_key, timeout=30.0)]
        agent._client_cycle = itertools.cycle(agent._clients)
        agent.client = agent._clients[0]

        # Initialize optimization history and iterations

        agent.action_functions = [
            {
                "type": "function",
                "function": {
                    "name": "action_display_analysis",
                    "description": "Summarize the latest evaluation failures into a structured failure taxonomy (extraction/runtime, format/constraint, reasoning/content) and combine with your own analysis to produce actionable evolution guidance. Call this after action_evaluate_on_task to get a ranked breakdown of failure modes, targeted fix recommendations, and an evolution strategy hint (prompt-tune vs structural edit).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "Your own analysis of the current state: case studies of failing examples, hypotheses about root causes, and a plan for the next actions."
                            }
                        },
                        "required": ["analysis"],
                        "additionalProperties": False,
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_read_logic",
                    "description": "Reads the source code of the specified logic (function, method, or class) within a given module.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to read. If the target_name contains a dot, it refers to a method within a class (e.g., 'Agent.action_call_llm')."
                            }
                        },
                        "required": ["module_name", "target_name"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_adjust_logic",
                    "description": "Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to modify/add/delete. If the target_name contains a dot, it refers to a method within a class."
                            },
                            "new_code": {
                                "type": "string",
                                "description": "The new logic as a string. (Ensure there is no extra indentation in new_code)"
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["function", "class"],
                                "description": "The type of target."
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["modify", "add", "delete"],
                                "description": "The operation to perform."
                            }
                        },
                        "required": ["module_name", "target_name", "new_code", "target_type", "operation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_run_code",
                    "description": "Execute Python or shell code and capture the output, errors, and return value. (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_type": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "description": "The type of code to execute."
                            },
                            "code": {
                                "type": "string",
                                "description": "The code to execute as a string."
                            }
                        },
                        "required": ["code_type", "code"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_call_json_format_llm",
                    "description": "Call an external LLM for assistance with gathering insights, refining strategies, correcting errors, and solving complex problems. Output response in JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "ID of the model to use."
                            },
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "enum": ["system", "assistant", "user"]
                                        },
                                        "content": {"type": "string"}
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False
                                },
                                "description": "A list of messages comprising the conversation so far."
                            },
                            "temperature": {
                                "type": "number",
                                "description": "What sampling temperature to use. Higher values will make the output more random, while lower values will make it more focused and deterministic."
                            },
                            "role": {
                                "type": "string",
                                "description": "The role that LLM play."
                            },
                            "return_dict_keys": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "An array containing the names of the keys that should be present in the returned dictionary."
                            },
                            "requirements": {
                                "type": "string",
                                "description": "A string that specifies the conditions required to perform a call to the LLM."
                            }
                        },
                        "required": ["model", "messages", "temperature", "role", "return_dict_keys", "requirements"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_select_examples",
                    "description": "Select a representative subset of task examples for future train, valid, or evaluate calls. Useful for diverse coverage or focused low-cost experiments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "split": {
                                "type": "string",
                                "enum": ["train", "valid", "test", "evaluate", "in_domain"],
                                "description": "Source pool to sample from."
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of examples to select. Use 0 with persist_for to clear a stored selection."
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["diverse", "random", "head"],
                                "description": "How to choose examples from the source pool."
                            },
                            "persist_for": {
                                "type": "string",
                                "enum": ["train", "valid", "evaluate", "optimize", "none"],
                                "description": "Which future call path should consume this selection. `optimize` maps to train for train/in_domain splits, valid for valid split, and evaluate for test/evaluate splits."
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed used by random or diverse selection."
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_compare_variants",
                    "description": "Run an A/B comparison of two solver or prompt variants on the same sampled validation subset. Use this before committing to a structural change (action_adjust_logic) or a prompt update to verify the new variant actually improves accuracy. Each variant can override prompt_config and/or provide full solver_code. An empty dict means 'current solver'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "variant_a": {
                                "type": "object",
                                "description": "First variant spec. Keys: label (string), prompt_config (object with role/requirements/temperature), solver_code (string with full solver function source). All keys optional; empty dict = current solver.",
                                "properties": {
                                    "label": {"type": "string", "description": "Human-readable name for this variant."},
                                    "prompt_config": {
                                        "type": "object",
                                        "description": "Prompt config overrides (role, requirements, temperature).",
                                        "properties": {
                                            "role": {"type": "string"},
                                            "requirements": {"type": "string"},
                                            "temperature": {"type": "number"}
                                        },
                                        "additionalProperties": False
                                    },
                                    "solver_code": {"type": "string", "description": "Full Python source defining solver(agent, task)."}
                                },
                                "additionalProperties": False
                            },
                            "variant_b": {
                                "type": "object",
                                "description": "Second variant spec. Same schema as variant_a.",
                                "properties": {
                                    "label": {"type": "string", "description": "Human-readable name for this variant."},
                                    "prompt_config": {
                                        "type": "object",
                                        "description": "Prompt config overrides (role, requirements, temperature).",
                                        "properties": {
                                            "role": {"type": "string"},
                                            "requirements": {"type": "string"},
                                            "temperature": {"type": "number"}
                                        },
                                        "additionalProperties": False
                                    },
                                    "solver_code": {"type": "string", "description": "Full Python source defining solver(agent, task)."}
                                },
                                "additionalProperties": False
                            },
                            "sample_count": {
                                "type": "integer",
                                "description": "Number of validation examples to sample for comparison (default 20)."
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed for reproducible sampling (default 42)."
                            }
                        },
                        "required": ["variant_a", "variant_b"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_optimize_prompt_on_task",
                    "description": "Run the inner-loop prompt optimizer for the current task. It evolves role/requirements/temperature using minibatch feedback and updates prompt_config.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "iterations": {
                                "type": "integer",
                                "description": "Number of optimization iterations."
                            },
                            "train_size": {
                                "type": "integer",
                                "description": "Number of in-domain training samples per optimization run."
                            },
                            "valid_size": {
                                "type": "integer",
                                "description": "Number of in-domain validation samples per optimization run."
                            },
                            "candidates_per_iter": {
                                "type": "integer",
                                "description": "Number of candidate prompts to test in each iteration."
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Optional sampling seed override for train/valid example selection during this optimization call."
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_get_evolution_credit",
                    "description": "Return current evolution credit assignment state, including Delta_U_P, Delta_U_C, prompt saturation status, and thresholds.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_evaluate_on_task",
                    "description": "Evaluate the current solver on the goal task samples and return the evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

        agent.disable_tool_calls = str(os.getenv("DISABLE_TOOL_CALLS", "0")).lower() in {"1", "true", "yes"}
        agent._reset_runtime_state()

    def _make_credit_state(agent):
        return {
            "last_prompt_delta": 0.0,
            "last_prompt_before": 0.0,
            "last_prompt_after": 0.0,
            "last_prompt_report": None,
            "last_failure_reflection": None,
            "prompt_saturated": False,
            "pending_structure_eval": False,
            "structure_eval_baseline": None,
            "last_structure_delta": None,
            "last_eval_acc": None,
            "inner_steps": 0,
        }

    def _reset_runtime_state(agent):
        agent.optimize_history = []
        agent.selected_examples = {}
        agent.credit_state = agent._make_credit_state()

    def _continue_or_stop_outer_loop(agent):
        print(f"========== Action Counter:  ==========\n{action_counter}", end='\n\n')
        if action_counter["evolve"] >= agent.max_outer_evolve_steps:
            print(
                "========== Outer Loop Stop ==========\n"
                f"Reached max_outer_evolve_steps: {action_counter['evolve']} / {agent.max_outer_evolve_steps}\n"
            )
            sys.exit(1)
        print(f"========== Agent Evolve:  ==========", end="\n\n")
        agent.evolve()

    def reinit(agent):
        agent._reset_runtime_state()
        first_aware_content = action_environment_aware(agent)
        solver_logic = action_read_logic("agent_module", "solver")
        solver_lines = len([line for line in solver_logic.splitlines() if line.strip()])
        print(
            "========== Startup Summary: ==========\n"
            f"task={agent.task_name}\n"
            f"infer_model={agent.default_infer_model}\n"
            f"optimizer_model={agent.optimization_model}\n"
            f"outer_loop_model={agent.outer_loop_model}\n"
            f"max_outer_evolve_steps={agent.max_outer_evolve_steps}\n"
            f"inner_loop_defaults={agent.inner_loop_defaults}\n"
            f"prompt_saturation_epsilon={agent.prompt_saturation_epsilon}\n"
            f"ab_gate_delta_threshold={agent.ab_gate_delta_threshold}\n"
            f"solver_logic_lines={solver_lines}\n",
            end="\n",
        )

        agent.optimize_history.append({"role": "user", "content": "The logic of solver:\n" + solver_logic})

    def _fallback_evolution_cycle(agent):
        """
        Fallback path for providers that do not support native tool calling.
        Runs a simple fused loop: inner prompt optimization then task evaluation.
        """
        print("========== Fallback Evolution (No Native Tools): ==========\n")
        total_steps = 2
        pbar = tqdm(total=total_steps, desc="Fallback evolve", unit="step") if tqdm is not None else None

        try:
            step_start = time.time()
            if pbar is not None:
                pbar.set_postfix_str("optimize prompt")
            opt_result = action_optimize_prompt_on_task(agent)
            agent.optimize_history.append({
                "role": "tool",
                "content": opt_result,
                "tool_call_id": f"fallback_opt_{action_counter['evolve']}",
            })
            elapsed = time.time() - step_start
            print(f"========== Fallback Prompt Optimization Result ({elapsed:.1f}s): ==========\n{opt_result}\n")
        except Exception:
            opt_result = _format_exception()
            agent.optimize_history.append({
                "role": "tool",
                "content": opt_result,
                "tool_call_id": f"fallback_opt_err_{action_counter['evolve']}",
            })
        finally:
            if pbar is not None:
                pbar.update(1)

        try:
            step_start = time.time()
            if pbar is not None:
                pbar.set_postfix_str("evaluate task")
            eval_result = action_evaluate_on_task(agent, agent.goal_task, functools.partial(solver, agent))
            agent.optimize_history.append({
                "role": "tool",
                "content": eval_result,
                "tool_call_id": f"fallback_eval_{action_counter['evolve']}",
            })
            elapsed = time.time() - step_start
            print(f"========== Fallback Evaluation Result ({elapsed:.1f}s): ==========\n{eval_result[:500]}...\n")
        except Exception:
            eval_result = _format_exception()
            print(f"========== Fallback Evaluation Error: ==========\n{eval_result}\n")
            agent.optimize_history.append({
                "role": "tool",
                "content": eval_result,
                "tool_call_id": f"fallback_eval_err_{action_counter['evolve']}",
            })
        finally:
            if pbar is not None:
                pbar.update(1)
                pbar.close()

        if agent.credit_state.get("prompt_saturated", False):
            current_round = int(action_counter["evolve"])
            min_rounds = int(agent.fallback_min_prompt_update_rounds)
            if current_round < min_rounds:
                print(
                    "========== Prompt Saturation Deferred ==========\n"
                    f"Current fallback round {current_round}/{min_rounds}. "
                    "Enforcing additional prompt-update rounds before stopping.\n"
                )
            else:
                print(
                    "========== Prompt Saturation Reached ==========\n"
                    "Fallback mode has no structural-edit tool calls, so prompt-only evolution stops here.\n"
                )
                return

        agent._continue_or_stop_outer_loop()

    def execute_action(agent, actions: typing.Dict):
        """
        Executes the function called by the model and returns the result.
        """
        is_reinit = False
        for tool_call in actions.tool_calls:
            print(f"========== Tool Call:  ==========\n{tool_call}", end="\n\n")
            tool_name = tool_call.function.name
            try:
                action_counter[tool_name] += 1
                raw_args = tool_call.function.arguments
                if not raw_args:
                    arguments = {}
                else:
                    try:
                        arguments = json.loads(raw_args, strict=False)
                    except json.JSONDecodeError:
                        # Attempt repair: extract first JSON object from malformed output
                        extracted = _extract_first_json_object(raw_args)
                        if extracted is not None:
                            try:
                                arguments = json.loads(extracted)
                            except json.JSONDecodeError:
                                # Try fixing invalid escape sequences
                                try:
                                    arguments = json.loads(_repair_json_escapes(extracted))
                                except json.JSONDecodeError:
                                    raise
                        else:
                            # Try fixing invalid escape sequences on raw args
                            try:
                                arguments = json.loads(_repair_json_escapes(raw_args))
                            except json.JSONDecodeError:
                                raise
                if tool_name == "action_display_analysis":
                    result = action_display_analysis(agent, **arguments)

                elif tool_name == "action_environment_aware":
                    result = action_environment_aware(agent, **arguments)

                elif tool_name == "action_read_logic":
                    result = action_read_logic(**arguments)

                elif tool_name == "action_adjust_logic":
                    if not agent.credit_state.get("prompt_saturated", False):
                        result = (
                            "Blocked structural update. Inner-loop prompt gains are not saturated yet.\n"
                            f"Need Delta_U_P < epsilon ({agent.prompt_saturation_epsilon:.4f}). "
                            f"Current Delta_U_P={float(agent.credit_state.get('last_prompt_delta', 0.0)):.4f}.\n"
                            "Run action_optimize_prompt_on_task and check action_get_evolution_credit first."
                        )
                    else:
                        require_ab_gate = (
                            arguments.get("module_name") == "agent_module"
                            and arguments.get("target_name") == "solver"
                            and arguments.get("target_type") == "function"
                            and arguments.get("operation") in {"modify", "add"}
                        )
                        if require_ab_gate:
                            compare_seed = int(action_counter.get("action_adjust_logic", 0)) + 42
                            compare_metrics = _compare_variant_metrics(
                                agent=agent,
                                variant_a={"label": "current"},
                                variant_b={
                                    "label": "candidate",
                                    "solver_code": arguments.get("new_code", ""),
                                },
                                sample_count=12,
                                seed=compare_seed,
                            )
                            compare_report = _format_compare_report(compare_metrics)
                            if compare_metrics["delta"] <= agent.ab_gate_delta_threshold:
                                result = (
                                    "Blocked structural update. Candidate solver did not beat the current solver "
                                    "on the required cheap A/B check.\n\n"
                                    f"{compare_report}"
                                )
                            else:
                                result = action_adjust_logic(**arguments)
                        else:
                            result = action_adjust_logic(**arguments)
                        baseline = agent.credit_state.get("last_eval_acc")
                        if baseline is not None and "Successfully" in str(result):
                            agent.credit_state["structure_eval_baseline"] = float(baseline)
                        agent.credit_state["pending_structure_eval"] = "Successfully" in str(result)

                elif tool_name == "action_run_code":
                    result = action_run_code(**arguments, agent=agent)
                    if arguments.get("code_type", None) == "python" and "self_evolving_agent.reinit()" in arguments.get("code", ""):
                        is_reinit = True

                elif tool_name == 'action_call_json_format_llm':
                    result = agent.action_call_json_format_llm(**arguments)
                    try:
                        print(f"========== Action Call JSON Format LLM Result:  ==========\n{json.loads(result[0])}", end="\n\n")
                    except:
                        print(f"========== Action Call JSON Format LLM Result:  ==========\n{result[0]}", end="\n\n")

                elif tool_name == "action_select_examples":
                    result = action_select_examples(agent, **arguments)
                elif tool_name == "action_compare_variants":
                    result = action_compare_variants(agent, **arguments)
                elif tool_name == "action_evaluate_on_task":
                    result = action_evaluate_on_task(agent, agent.goal_task, functools.partial(solver, agent))
                elif tool_name == "action_optimize_prompt_on_task":
                    result = action_optimize_prompt_on_task(agent, **arguments)
                elif tool_name == "action_get_evolution_credit":
                    result = action_get_evolution_credit(agent)
                else:
                    raise ValueError(f"Unknown function name: {tool_name}")

            except Exception:
                action_counter["error_handle"] += 1
                result = _format_exception()

            summarized_result = _summarize_tool_result(tool_name, result)
            print(f"========== Tool Call Result:  ==========\n{summarized_result}", sep="", end="\n\n")
            if is_reinit:
                break
            agent.optimize_history.append({"role": "tool", 
                                           "content": result, 
                                            "tool_call_id": tool_call.id})

        agent._continue_or_stop_outer_loop()

    def evolve(agent):
        """
        Evolves the agent by prompting the LLM to suggest improvements.
        """
        print('-' * 120)
        action_counter["evolve"] += 1

        tool_call_ids = set()
        remain_optimize_history = []
        for message in agent.optimize_history[-10:]:
            if not isinstance(message, dict) and message.role == "assistant" and message.tool_calls:
                tool_call_ids = set()
                for tool_call in message.tool_calls:
                    tool_call_ids.add(tool_call.id)
            if not isinstance(message, dict) and message.role == "tool" and message.tool_call_id not in tool_call_ids:
                print(f"========== Pop Item:  ==========\n{message}", end='\n\n')
                continue
            remain_optimize_history.append(message)
        agent.optimize_history = remain_optimize_history

        messages = [{"role": "system", "name": "Principles", "content": agent.goal_prompt}, 
                    {"role": "system", "name": "Environment", "content": action_environment_aware(agent)},
                    *agent.optimize_history,
                    {"role": "user", "content": "Analyze the current optimization state and call the next tool that most improves the agent."}]
        if agent.disable_tool_calls:
            agent._fallback_evolution_cycle()
            return
        try:
            response = agent.action_call_llm(messages=messages, model=agent.outer_loop_model, response_format="text", tools=agent.action_functions, tool_choice="auto")
        except Exception as e:
            error_text = repr(e)
            if (
                "tool-call-parser" in error_text
                or "tool_choice=\"required\"" in error_text
                or "Invalid JSON" in error_text
                or "Invalid \\\\escape" in error_text
                or "Invalid \\escape" in error_text
                or "validation error for list[function-wrap" in error_text
            ):
                print(
                    "========== Tool Calling Unsupported by Backend ==========\n"
                    "Backend rejected the native tool-call payload. "
                    "Switching to fallback evolution cycle without native tool calls.\n"
                )
                agent.disable_tool_calls = True
                agent._fallback_evolution_cycle()
                return
            print(error_text)
            for message in messages:
                print(f"========== Message:  ==========\n{message}", end="\n\n")
            sys.exit(1)

        # Sanitize tool call arguments to fix invalid JSON escapes from
        # models like Mistral before storing in history.  This prevents
        # BadRequestError on subsequent evolve() calls when the history
        # is sent back to the vLLM server.
        resp_msg = response[0]
        if getattr(resp_msg, "tool_calls", None):
            for tc in resp_msg.tool_calls:
                raw = tc.function.arguments
                if raw:
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError:
                        repaired = _repair_json_escapes(raw)
                        try:
                            json.loads(repaired)
                            tc.function.arguments = repaired
                        except json.JSONDecodeError:
                            pass

        agent.optimize_history.append(resp_msg)
        agent.execute_action(resp_msg)

    def _estimate_message_tokens(agent, message):
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content", ""))
        else:
            content = str(getattr(message, "content", ""))
        return max(1, len(content) // 4)

    def _trim_messages_to_budget(agent, messages, reserved_output_tokens=None):
        reserved_output_tokens = agent.max_output_tokens if reserved_output_tokens is None else int(reserved_output_tokens)
        max_prompt_tokens = max(1024, int(agent.max_input_tokens) - max(0, reserved_output_tokens))
        trimmed = list(messages)

        def total_tokens(items):
            return sum(agent._estimate_message_tokens(item) for item in items)

        def _get_tool_call_ids(msg):
            """Extract tool_call ids from an assistant message."""
            tc = None
            if isinstance(msg, dict):
                tc = msg.get("tool_calls")
            else:
                tc = getattr(msg, "tool_calls", None)
            if not tc:
                return set()
            return {getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None) for c in tc} - {None}

        def _get_tool_call_id(msg):
            """Extract tool_call_id from a tool-role message."""
            if isinstance(msg, dict):
                return msg.get("tool_call_id")
            return getattr(msg, "tool_call_id", None)

        while len(trimmed) > 2 and total_tokens(trimmed) > max_prompt_tokens:
            removed = False
            for idx in range(2, len(trimmed)):
                candidate = trimmed[idx]
                role = candidate.get("role") if isinstance(candidate, dict) else getattr(candidate, "role", None)
                if role == "assistant":
                    # Remove this assistant message and any orphaned tool results
                    tc_ids = _get_tool_call_ids(candidate)
                    trimmed.pop(idx)
                    if tc_ids:
                        trimmed = [m for i, m in enumerate(trimmed)
                                   if i < idx or _get_tool_call_id(m) not in tc_ids]
                    removed = True
                    break
                elif role == "tool":
                    # Remove this tool message and its parent assistant tool_call if it
                    # would become orphaned (i.e. all its tool results are gone).
                    orphan_tc_id = _get_tool_call_id(candidate)
                    trimmed.pop(idx)
                    if orphan_tc_id:
                        # Check if the parent assistant message still has other tool results
                        remaining_tc_ids = {_get_tool_call_id(m) for m in trimmed
                                            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "tool"}
                        for aidx in range(len(trimmed)):
                            am = trimmed[aidx]
                            a_role = am.get("role") if isinstance(am, dict) else getattr(am, "role", None)
                            if a_role == "assistant":
                                a_tc_ids = _get_tool_call_ids(am)
                                if orphan_tc_id in a_tc_ids and not (a_tc_ids & remaining_tc_ids):
                                    trimmed.pop(aidx)
                                    break
                    removed = True
                    break
            if not removed:
                trimmed.pop(2)
        return trimmed

    def _normalize_messages_for_backend(agent, messages):
        normalized = []
        system_chunks = []

        for raw_message in messages:
            if isinstance(raw_message, dict):
                message = dict(raw_message)
            else:
                message = {
                    "role": getattr(raw_message, "role", ""),
                    "content": getattr(raw_message, "content", ""),
                }
                if getattr(raw_message, "tool_calls", None) is not None:
                    message["tool_calls"] = raw_message.tool_calls
                if getattr(raw_message, "tool_call_id", None) is not None:
                    message["tool_call_id"] = raw_message.tool_call_id
                if getattr(raw_message, "name", None) is not None:
                    message["name"] = raw_message.name

            message["content"] = str(message.get("content", ""))
            # Sanitize control characters that break JSON serialization in
            # vLLM / OpenAI request payloads.  Replace chars 0x00-0x1F
            # (except \n \r \t which are common and safe) with spaces.
            _content = message["content"]
            message["content"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', _content)
            role = message.get("role")

            if role == "system":
                name = message.get("name")
                content = message["content"].strip()
                if content:
                    system_chunks.append(f"[{name}]\n{content}" if name else content)
                continue

            normalized.append(message)

        if system_chunks:
            normalized.insert(0, {"role": "system", "content": "\n\n".join(system_chunks)})

        # Mistral models require an assistant message between tool results and
        # the next user turn.  Insert a lightweight bridging message whenever a
        # "user" message directly follows a "tool" message.
        # Also ensure a "tool" message never directly follows "system" — insert
        # a bridging assistant message in that case too.
        # First, collect all tool_call ids present in assistant messages so we
        # can drop orphaned tool-role messages whose tool_call_id has no match.
        _present_tc_ids = set()
        for msg in normalized:
            if msg.get("role") == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    for c in tc:
                        cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                        if cid:
                            _present_tc_ids.add(cid)
        # Drop tool messages whose tool_call_id is not in any assistant message
        normalized = [
            msg for msg in normalized
            if msg.get("role") != "tool" or msg.get("tool_call_id") in _present_tc_ids
        ]
        # Also collect all tool_call_ids that have a tool-role response present
        _responded_tc_ids = {msg.get("tool_call_id") for msg in normalized if msg.get("role") == "tool"} - {None}
        # Drop assistant messages whose tool_calls have NO matching tool results
        # (they would confuse the backend expecting tool results to follow)
        _cleaned = []
        for msg in normalized:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"]
                tc_ids = set()
                for c in tc:
                    cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                    if cid:
                        tc_ids.add(cid)
                if tc_ids and not (tc_ids & _responded_tc_ids):
                    # No tool results for any of this assistant's tool_calls — drop it
                    continue
            _cleaned.append(msg)
        normalized = _cleaned

        fixed: typing.List[dict] = []
        for msg in normalized:
            prev_role = fixed[-1].get("role") if fixed else None
            cur_role = msg.get("role")
            if cur_role == "tool" and prev_role in ("system", "user", None):
                fixed.append({"role": "assistant", "content": "Understood."})
            elif cur_role == "user" and prev_role == "tool":
                fixed.append({"role": "assistant", "content": "Understood."})
            fixed.append(msg)

        return fixed

    def action_call_json_format_llm(
        agent,
        *,
        messages: typing.List[typing.Dict[str, str]], 
        model: str = "gpt-4o-mini",
        temperature: float = 1.0, 
        # max_completion_tokens: int = 4096, 
        num_of_response: int = 1,
        role: str = "task solver", 
        return_dict_keys: typing.List[str] = [], 
        requirements: str = "", 
    ):
        system_prompt = (
            f"You are a helpful {role}.\n"
            f"Reply in JSON format, ONLY using the keys {return_dict_keys}.\n"
            f"Requirements:\n{requirements}"
        ).strip()
        _messages = [{"role": "system", "content": system_prompt}, *messages]
        return_dicts = agent.action_call_llm(model=model,
                                    messages=_messages, 
                                    temperature=temperature,
                                    # max_completion_tokens=max_completion_tokens,
                                    n=num_of_response,
                                    response_format="json")
        
        for key in return_dict_keys:
            for return_dict in return_dicts:
                if key not in return_dict:
                    return_dict[key] = f"NO {key} IN DICTIONARY"
        return return_dicts
    
    def action_call_llm(
        agent, 
        *,
        model: str = "gpt-4o-mini",
        messages: typing.List[typing.Dict[str, str]], 
        temperature: float = 1.0, 
        # max_completion_tokens: int = 4096, 
        n: int = 1,
        response_format: typing.Literal["text", "json", "json_object"] = "text", 
        tools=None, 
        tool_choice=None,
    ):
        """
        Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

        Args:
            agent (Agent): The OpenAI client instance used to interact with the LLM.
            messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
            response_format (str): The desired format of the LLM's output.
            model (str): Specifies which LLM model to use.
            temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
            max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
            n (int): The number of chat completion choices to generate for each input message.

        Returns:
            response (dict): The response from the OpenAI LLM.
        """
        def try_parse_json(content):
            try:
                return json.loads(content)
            except Exception:
                extracted = _extract_first_json_object(content)
                if extracted is not None:
                    try:
                        return json.loads(extracted)
                    except Exception:
                        pass
                return {"JSONDecodeError": content}

        if response_format == "json":
            response_format = "json_object"

        messages = agent._normalize_messages_for_backend(messages)

        kwargs = {
            "n": n,
            "model": model,
            "messages": agent._trim_messages_to_budget(messages),
            "temperature": temperature,
            "max_tokens": agent.max_output_tokens,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        last_err = None
        max_retries = len(agent._clients) * 2  # allow retries for transient parse errors
        for _attempt in range(max_retries):
            client = next(agent._client_cycle)
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except openai.APIConnectionError as e:
                last_err = e
                print(f"[llm] Connection failed for {client.base_url}, trying next server...", flush=True)
            except openai.BadRequestError as e:
                last_err = e
                err_msg = str(e)
                # Retry on JSON parse failures from the tool-call parser — the model
                # may produce valid output on the next attempt.
                if "delimiter" in err_msg or "Expecting" in err_msg or "Invalid JSON" in err_msg or "control character" in err_msg:
                    print(f"[llm] Bad JSON in model output (attempt {_attempt+1}/{max_retries}), retrying...", flush=True)
                else:
                    raise
        else:
            raise last_err

        if response_format == "text":
            return [choice.message for choice in response.choices]
        return [try_parse_json(choice.message.content) for choice in response.choices]
