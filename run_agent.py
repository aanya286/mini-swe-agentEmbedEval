"""
run_agent.py

Runs mini-swe-agent's DefaultAgent with DockerEnvironment to evaluate
an LLM's ability to fix an embedded systems PR inside a Zephyr/QEMU container.

Usage:
    python run_agent.py \
        --image ghcr.io/yourproject/zephyr-embedbench:latest \
        --model claude-sonnet-4-20250514 \
        --diff path/to/pr.diff \
        --spec "Fix the off-by-one error in the ring buffer implementation" \
        --test-cmd "west build -b native_posix && ./build/zephyr/zephyr.exe"
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
You are an expert embedded systems engineer fixing bugs in Zephyr RTOS firmware.
You have bash access to a Docker container with Zephyr, west, and QEMU installed.

Your only tool is bash. To run a command, output it in a bash block:
```bash
<your command here>
```

Rules:
- Fix the issue described in the task.
- Use the provided test command to verify your fix.
- When tests pass, output EXACTLY this as your final command (returncode must be 0):
```bash
echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
echo "Fixed: <one line description of what you changed>"
```
- Do not modify test files.
- Do not guess — read the code, understand the bug, then fix it.
"""

INSTANCE_TEMPLATE = """\
## Task
{{ task }}

## PR Diff (context — shows what was broken/changed)
```diff
{{ diff }}
```

## Test command to verify your fix
```
{{ test_cmd }}
```

Start by exploring the relevant files, then implement your fix.
"""


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run(
    image: str,
    model_name: str,
    diff: str,
    spec: str,
    test_cmd: str,
    workdir: str = "/workspace",
    step_limit: int = 30,
    cost_limit: float = 3.0,
    output_path: str | None = None,
) -> dict:
    """
    Run the agent on a single PR instance.

    Returns:
        Dict with 'exit_status' and 'submission' keys from the agent.
    """
    env = DockerEnvironment(image=image, cwd=workdir)
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
        task=spec,
        diff=diff,
        test_cmd=test_cmd,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run EmbedBench mini-swe-agent evaluation")
    parser.add_argument("--image", required=True, help="Docker image name (from Rishi)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LiteLLM model string")
    parser.add_argument("--diff", required=True, help="Path to PR diff file (or raw diff string)")
    parser.add_argument("--spec", required=True, help="Natural language description of the issue to fix")
    parser.add_argument("--test-cmd", required=True, help="Command to run inside container to verify fix")
    parser.add_argument("--workdir", default="/workspace", help="Working directory inside container")
    parser.add_argument("--step-limit", type=int, default=30, help="Max agent steps")
    parser.add_argument("--cost-limit", type=float, default=3.0, help="Max USD cost")
    parser.add_argument("--output", help="Save trajectory JSON to this path")
    args = parser.parse_args()

    # Load diff from file if it's a path, otherwise treat as raw string
    diff_path = Path(args.diff)
    diff = diff_path.read_text() if diff_path.exists() else args.diff

    result = run(
        image=args.image,
        model_name=args.model,
        diff=diff,
        spec=args.spec,
        test_cmd=args.test_cmd,
        workdir=args.workdir,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        output_path=args.output,
    )

    print(json.dumps(result, indent=2))

    # Exit 0 if submitted (tests passed), 1 otherwise
    sys.exit(0 if result.get("exit_status") == "Submitted" else 1)


if __name__ == "__main__":
    main()