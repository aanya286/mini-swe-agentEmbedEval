"""
run_agent.py

Runs mini-swe-agent's DefaultAgent with DockerEnvironment to evaluate
an LLM's ability to fix an embedded systems PR inside a Zephyr/QEMU container.

Usage:
    python run_agent.py \
        --instance zephyr__zephyr-65697 \
        --model claude-sonnet-4-20250514

Or with overrides:
    python run_agent.py \
        --instance zephyr__zephyr-65697 \
        --model gpt-4o \
        --step-limit 50 \
        --output trajectories/zephyr-65697.json
"""

import argparse
import json
import sys
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.docker import DockerEnvironment


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are an expert embedded systems engineer. You can interact with a
Linux shell to navigate codebases, edit source files, build firmware,
and run tests. You are working inside a Zephyr RTOS repository.

Your response must contain exactly ONE bash code block with ONE command
(or commands connected with && or ||). Include a THOUGHT section before
your command explaining your reasoning.

<format_example>
THOUGHT: Your reasoning here
```bash
your_command_here
```
</format_example>

## Zephyr Developer Guide

**Build system**: Zephyr uses `west` + CMake + Ninja.
- Build: `west build -b <board> <test_path>`
- Run tests: `west build -t run` (runs on QEMU, must build first)
- Clean rebuild: `rm -rf build && west build -b <board> <test_path>`
- Incremental rebuild after editing a .c file: just run `west build` again (fast)

**Directory structure**:
- `kernel/` — Core kernel
- `lib/posix/` — POSIX API implementation
- `subsys/` — Subsystems
- `drivers/` — Device drivers
- `tests/` — Test suites
- `include/` — Public headers

**Test framework**: Zephyr uses `ztest`. Tests defined with `ZTEST(suite, test_name)`.
Assertions: `zassert_ok()`, `zassert_equal()`, `zassert_true()`, etc.

**Code navigation**:
- `grep -rn "function_name" --include="*.c" --include="*.h" .`
- `grep "function_name" tags` (ctags index pre-built at /testbed/tags)

**When done**, submit with:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

INSTANCE_TEMPLATE = """\
<issue_description>
{{ task }}
</issue_description>

<project_context>
Project: Zephyr RTOS
Repository is at: /testbed
</project_context>

<test_info>
The following tests are currently FAILING and should pass after your fix:
{{ fail_to_pass }}

Tests that must continue to pass:
{{ pass_to_pass }}

Build command: {{ build_command }}
Run tests: {{ run_command }}
Test directory: {{ test_path }}
Target platform: {{ platform }}
</test_info>

<instructions>
Fix the source code so the failing tests pass. Do NOT modify test files.

Recommended workflow:
1. Read the failing test code to understand what behavior is expected
2. Use grep/ctags to find the relevant source code
3. Understand the bug
4. Edit the source code to fix it
5. Rebuild and run tests to verify
6. Submit: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
</instructions>
"""


# ── Runner ────────────────────────────────────────────────────────────────────

def load_metadata(instance_id: str) -> dict:
    """Load instance metadata.json from the standard location."""
    meta_path = Path(f"docker/instances/{instance_id}/metadata.json")
    if not meta_path.exists():
        # fallback: look next to this script
        meta_path = Path(__file__).parent / "metadata.json"
    return json.loads(meta_path.read_text())


def run(
    instance_id: str,
    model_name: str,
    step_limit: int = 100,
    cost_limit: float = 5.0,
    timeout: int = 180,
    output_path: str | None = None,
) -> dict:
    """
    Run the agent on a single benchmark instance.

    Returns:
        Dict with 'exit_status' and 'submission' keys.
    """
    meta = load_metadata(instance_id)

    print(f"[*] Instance:  {instance_id}")
    print(f"[*] Image:     {meta['docker_image']}")
    print(f"[*] Model:     {model_name}")
    print(f"[*] Bug:       {meta['problem_statement'][:80]}...")

   
    env = DockerEnvironment(
        image=meta["docker_image"],
        cwd="/testbed",                  # all Zephyr source lives here
        timeout=timeout,                 # west build can take ~60-120s
        forward_env=["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"],
    )

    model = LitellmModel(model_name=model_name)

    agent = DefaultAgent(
        model,
        env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        step_limit=step_limit,
        cost_limit=cost_limit,
        output_path=Path(output_path) if output_path else None,
    )

    result = agent.run(
        task=meta["problem_statement"],
        platform=meta["platform"],
        test_path=meta["test_path"],
        build_command=meta["build_command"],
        run_command=meta["run_command"],
        fail_to_pass=", ".join(meta["fail_to_pass"]),
        pass_to_pass=", ".join(meta["pass_to_pass"]),
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run EmbedBench mini-swe-agent evaluation")
    parser.add_argument("--instance", default="zephyr__zephyr-65697", help="Instance ID")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LiteLLM model string")
    parser.add_argument("--step-limit", type=int, default=100, help="Max agent steps")
    parser.add_argument("--cost-limit", type=float, default=5.0, help="Max USD cost")
    parser.add_argument("--timeout", type=int, default=180, help="Per-command timeout in seconds")
    parser.add_argument("--output", help="Save trajectory JSON to this path")
    args = parser.parse_args()

    result = run(
        instance_id=args.instance,
        model_name=args.model,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        timeout=args.timeout,
        output_path=args.output,
    )

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("exit_status") == "Submitted" else 1)


if __name__ == "__main__":
    main()