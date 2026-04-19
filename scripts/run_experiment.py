"""
BitsAndBytes experiment runner. Usage:
    python scripts/run_experiment.py scripts/configs/llama32_3b.yaml
"""
from common import load_config, load_all_prompts, build_messages
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from pathlib import Path
from datetime import datetime
import json
import argparse
from common import ensure_utf8
ensure_utf8()


def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def generate_response(model, tokenizer, messages, max_new_tokens=256):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def run_config(model, tokenizer, prompts, config_label):
    responses = []
    n = len(prompts)
    for i, p in enumerate(prompts):
        goal = p["meta"]["goal"]
        print(f"  [{i + 1}/{n}] {goal[:80]}")
        try:
            prompt_text = p.get("prompt_text") or goal
            messages = build_messages(prompt_text)
            response = generate_response(model, tokenizer, messages)
        except Exception as e:
            print(f"    ERROR ({config_label}): {e}")
            response = None
        responses.append(response)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    num_prompts = cfg["num_prompts"]
    results_subdir = cfg["results_subdir"]
    experiment_prefix = cfg["experiment_prefix"]

    results_dir = Path("results_reformatted") / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = get_compute_dtype()

    print(f"Loading {num_prompts} prompts from jailbreakbench...")
    prompts = load_all_prompts(num_prompts)

    # --- Baseline ---
    print(f"\n=== Baseline ({compute_dtype}) ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    baseline_responses = run_config(model, tokenizer, prompts, "baseline")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 8-bit quantized ---
    print(f"\n=== 8-bit INT8 ({compute_dtype} compute) ===")
    quant_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config_8bit,
    )
    quant_8bit_responses = run_config(
        model, tokenizer, prompts, "quantized_8bit")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 4-bit quantized ---
    print(f"\n=== 4-bit NF4 ({compute_dtype} compute) ===")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config,
    )
    quant_responses = run_config(model, tokenizer, prompts, "quantized_4bit")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Assemble results ---
    results = []
    for p, baseline_out, quant_8bit_out, quant_out in zip(
        prompts, baseline_responses, quant_8bit_responses, quant_responses
    ):
        results.append({
            "prompt_index": p["meta"]["artifact_index"],
            "goal": p["meta"]["goal"],
            "behavior": p["meta"]["behavior"],
            "category": p["meta"]["category"],
            "outputs": {
                "baseline": baseline_out,
                "quantized_8bit": quant_8bit_out,
                "quantized_4bit": quant_out,
            },
            "manual_labels": {
                variant: {
                    "willingness": None,
                    "prompt_alignment": None,
                    "details": None,
                }
                for variant in ("baseline", "quantized_8bit", "quantized_4bit")
            },
        })

    output = {
        "experiment_name": f"{experiment_prefix}_baseline_vs_8bit_vs_4bit_pair{num_prompts}",
        "base_model": model_id,
        "artifact": {
            "method": "PAIR",
            "source_model": "vicuna-13b-v1.5",
        },
        "settings": {
            "baseline_dtype": str(compute_dtype),
            "quantized_8bit_mode": "bitsandbytes_8bit_int8",
            "quantized_4bit_mode": "bitsandbytes_4bit_nf4",
            "quantized_compute_dtype": str(compute_dtype),
            "max_new_tokens": 256,
            "do_sample": False,
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
