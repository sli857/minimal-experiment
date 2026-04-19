"""
GGUF experiment runner. Usage:
    python scripts/run_gguf_experiment.py scripts/configs/llama32_3b_gguf.yaml
"""
from common import load_config, load_all_prompts, build_messages
from llama_cpp import Llama
from pathlib import Path
from datetime import datetime
import json
import argparse
from common import ensure_utf8
ensure_utf8()


def generate_response_gguf(llm, messages, max_tokens=256):
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return response["choices"][0]["message"]["content"].strip()


def run_gguf_config(llm, prompts, config_label):
    responses = []
    n = len(prompts)
    for i, p in enumerate(prompts):
        goal = p["meta"]["goal"]
        print(f"  [{i + 1}/{n}] {goal[:80]}")
        try:
            prompt_text = p.get("prompt_text") or goal
            messages = build_messages(prompt_text)
            response = generate_response_gguf(llm, messages)
        except Exception as e:
            print(f"    ERROR ({config_label}): {e}")
            response = None
        responses.append(response)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to GGUF YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg["model_name"]
    num_prompts = cfg["num_prompts"]
    results_subdir = cfg["results_subdir"]
    experiment_prefix = cfg["experiment_prefix"]
    gguf_variants = cfg["gguf_variants"]

    results_dir = Path("results_reformatted") / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {num_prompts} prompts from jailbreakbench...")
    prompts = load_all_prompts(num_prompts)

    all_responses = {}

    for variant in gguf_variants:
        label = variant["label"]
        gguf_path = variant["path"]
        print(f"\n=== GGUF: {label} ({gguf_path}) ===")

        llm = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )
        all_responses[label] = run_gguf_config(llm, prompts, label)
        del llm

    # --- Assemble results ---
    labels = [v["label"] for v in gguf_variants]
    results = []
    for i, p in enumerate(prompts):
        results.append({
            "prompt_index": p["meta"]["artifact_index"],
            "goal": p["meta"]["goal"],
            "behavior": p["meta"]["behavior"],
            "category": p["meta"]["category"],
            "outputs": {label: all_responses[label][i] for label in labels},
            "manual_labels": {
                label: {
                    "willingness": None,
                    "prompt_alignment": None,
                    "details": None,
                }
                for label in labels
            },
        })

    variant_names = "_vs_".join(labels)
    output = {
        "experiment_name": f"{experiment_prefix}_{variant_names}_pair{num_prompts}",
        "base_model": model_name,
        "artifact": {
            "method": "PAIR",
            "source_model": "vicuna-13b-v1.5",
        },
        "settings": {
            "backend": "llama-cpp-python",
            "gguf_variants": gguf_variants,
            "max_new_tokens": 256,
            "temperature": 0.0,
        },
        "notes": "manual_labels per variant: willingness, prompt_alignment, details = 1-5 ratings.",
        "results": results,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / \
        f"{experiment_prefix}_pair{num_prompts}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
