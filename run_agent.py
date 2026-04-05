"""
run_agent.py

Runs mini-swe-agent's DefaultAgent with DockerEnvironment to evaluate
an LLM's ability to fix an embedded systems PR inside a Zephyr/QEMU container.

Usage:
    python run_agent.py --instance zephyr__zephyr-65697 --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.docker import DockerEnvironment

# Path to the YAML config (same directory as this script)
CONFIG_PATH = Path(__file__).parent / "embedbench.yaml"



def load_metadata(instance_id: str, data_dir: Path) -> dict:
    """Load instance metadata.json from the EmbedEval repo."""
    meta_path = data_dir / "docker" / "instances" / instance_id / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}. "
            "Pass --data-dir pointing to the EmbedEval repo root."
        )
    return json.loads(meta_path.read_text())


def run(
    instance_id: str,
    data_dir: Path,
    model_name: str | None = None,
    timeout: int | None = None,
    output_path: str | None = None,
) -> dict:
    meta = load_metadata(instance_id, data_dir)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    agent_cfg = cfg.get("agent", {})
    model_cfg = cfg.get("model", {})
    env_cfg = {k: v for k, v in cfg.get("environment", {}).items() if k != "environment_class"}

    # CLI args override YAML values when provided
    if model_name is not None:
        model_cfg["model_name"] = model_name
    if timeout is not None:
        env_cfg["timeout"] = timeout
    if output_path is not None:
        agent_cfg["output_path"] = Path(output_path)

    print(f"[*] Instance: {instance_id}")
    print(f"[*] Image:    {meta['docker_image']}")
    print(f"[*] Model:    {model_cfg.get('model_name')}")

    env = DockerEnvironment(image=meta["docker_image"], **env_cfg)
    model = LitellmModel(**model_cfg)
    agent = DefaultAgent(model, env, **agent_cfg)

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
    parser.add_argument("--data-dir", default=None, help="Path to EmbedEval repo root (contains docker/instances/)")
    parser.add_argument("--model", default=None, help="LiteLLM model string (overrides embedbench.yaml)")
    parser.add_argument("--timeout", type=int, default=None, help="Per-command timeout in seconds (overrides embedbench.yaml)")
    parser.add_argument("--output", help="Save trajectory JSON to this path")
    args = parser.parse_args()

    # Resolve data_dir: CLI flag > EMBEDEVAL_DIR env var > sibling directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif "EMBEDEVAL_DIR" in os.environ:
        data_dir = Path(os.environ["EMBEDEVAL_DIR"])
    else:
        data_dir = Path(__file__).parent.parent / "EmbedEval"

    result = run(
        instance_id=args.instance,
        data_dir=data_dir,
        model_name=args.model,
        timeout=args.timeout,
        output_path=args.output,
    )

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("exit_status") == "Submitted" else 1)


if __name__ == "__main__":
    main()