import json
from pathlib import Path

import torch
import jailbreakbench as jbb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = Path("results/minimal")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_prompt():
    artifact = jbb.read_artifact(
        method="PAIR",
        model_name="vicuna-13b-v1.5",
    )
    jb = artifact.jailbreaks[0]
    return {
        "prompt_text": jb.prompt,
        "meta": {
            "artifact_method": "PAIR",
            "artifact_source_model": "vicuna-13b-v1.5",
            "artifact_index": jb.index,
            "goal": jb.goal,
            "behavior": jb.behavior,
            "category": jb.category,
        },
    }


def build_messages(user_prompt: str):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
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
    response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
    return response.strip()


def run_baseline(messages):
    compute_dtype = get_compute_dtype()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    output = generate_response(model, tokenizer, messages)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, str(compute_dtype)


def run_4bit(messages):
    compute_dtype = get_compute_dtype()
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
    output = generate_response(model, tokenizer, messages)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, str(compute_dtype)


def main():
    prompt_info = load_prompt()
    messages = build_messages(prompt_info["prompt_text"])

    baseline_output, baseline_dtype = run_baseline(messages)
    quant_output, quant_dtype = run_4bit(messages)

    result = {
        "experiment_name": "minimal_qwen25_bnb_baseline_vs_4bit",
        "base_model": MODEL_ID,
        "prompt_meta": prompt_info["meta"],
        "settings": {
            "baseline_dtype": baseline_dtype,
            "quantized_mode": "bitsandbytes_4bit_nf4",
            "quantized_compute_dtype": quant_dtype,
            "max_new_tokens": 256,
            "do_sample": False,
        },
        "outputs": {
            "baseline": baseline_output,
            "quantized_4bit": quant_output,
        },
        "manual_labels": {
            "baseline": None,
            "quantized_4bit": None,
        },
        "notes": "Fill manual labels as refusal / partial_compliance / full_compliance"
    }

    out_path = RESULTS_DIR / "qwen_bnb_run_001.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()