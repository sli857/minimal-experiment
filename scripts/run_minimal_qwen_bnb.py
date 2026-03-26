import json
from datetime import datetime
from pathlib import Path

import torch
import jailbreakbench as jbb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
NUM_PROMPTS = 10
RESULTS_DIR = Path("results/minimal")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_all_prompts(n):
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
            messages = build_messages(p["prompt_text"])
            response = generate_response(model, tokenizer, messages)
        except Exception as e:
            print(f"    ERROR ({config_label}): {e}")
            response = None
        responses.append(response)
    return responses


def main():
    compute_dtype = get_compute_dtype()

    print(f"Loading {NUM_PROMPTS} prompts from jailbreakbench...")
    prompts = load_all_prompts(NUM_PROMPTS)

    # --- Baseline ---
    print(f"\n=== Baseline ({compute_dtype}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    baseline_responses = run_config(model, tokenizer, prompts, "baseline")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quant_config,
    )
    quant_responses = run_config(model, tokenizer, prompts, "quantized_4bit")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Assemble results ---
    results = []
    for p, baseline_out, quant_out in zip(prompts, baseline_responses, quant_responses):
        results.append({
            "prompt_index": p["meta"]["artifact_index"],
            "goal": p["meta"]["goal"],
            "behavior": p["meta"]["behavior"],
            "category": p["meta"]["category"],
            "outputs": {
                "baseline": baseline_out,
                "quantized_4bit": quant_out,
            },
            "manual_labels": {
                "baseline": None,
                "quantized_4bit": None,
            },
        })

    output = {
        "experiment_name": f"qwen25_bnb_baseline_vs_4bit_pair{NUM_PROMPTS}",
        "base_model": MODEL_ID,
        "artifact": {
            "method": "PAIR",
            "source_model": "vicuna-13b-v1.5",
        },
        "settings": {
            "baseline_dtype": str(compute_dtype),
            "quantized_mode": "bitsandbytes_4bit_nf4",
            "quantized_compute_dtype": str(compute_dtype),
            "max_new_tokens": 256,
            "do_sample": False,
        },
        "notes": "Fill manual_labels as: refusal / partial_compliance / full_compliance",
        "results": results,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"qwen_bnb_pair{NUM_PROMPTS}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
