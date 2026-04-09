"""
Shared utilities for experiment runners.
"""
import sys
import os


def ensure_utf8():
    """Restart process with PYTHONUTF8=1 on Windows to avoid GBK encoding issues."""
    if not os.environ.get("PYTHONUTF8") and sys.flags.utf8_mode == 0:
        import subprocess
        result = subprocess.run(
            [sys.executable] + sys.argv,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        sys.exit(result.returncode)


def load_config(yaml_path: str) -> dict:
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_prompts(n):
    import jailbreakbench as jbb
    artifact = jbb.read_artifact(
        method="PAIR",
        model_name="vicuna-13b-v1.5",
    )
    prompts = []
    for jb in artifact.jailbreaks[:n]:
        prompts.append({
            "prompt_text": jb.prompt,
            "meta": {
                "artifact_method": "PAIR",
                "artifact_source_model": "vicuna-13b-v1.5",
                "artifact_index": jb.index,
                "goal": jb.goal,
                "behavior": jb.behavior,
                "category": jb.category,
            },
        })
    return prompts


def build_messages(user_prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
