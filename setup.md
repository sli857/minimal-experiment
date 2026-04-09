# LLM Safety Under Quantization

ECE 8803, Georgia Tech, Spring 2026.

Team Members: Shuyi Li, Jingyu Liu, Raj Patel, 
Abhinav Vemulapalli

Tests whether 4-bit BitsAndBytes (NF4) quantization affects the safety guardrails of instruction-tuned LLMs. Each model is run against PAIR jailbreak prompts under two conditions — full-precision baseline and 4-bit quantized — and the outputs are manually labeled.

---

## Directory Structure

```
.
├── envs/
│   ├── qwen/
│   │   ├── .venv/            # Qwen Python environment (gitignored)
│   │   └── requirements.txt  # Pinned dependencies
│   └── llama/
│       ├── .venv/            # Llama Python environment (gitignored)
│       └── requirements.txt
├── models/
│   └── hf/
│       ├── Qwen2.5-3B-Instruct/     # Downloaded model weights (gitignored)
│       └── Llama-3.2-3B-Instruct/
├── results/
│   ├── qwen/                 # Qwen experiment outputs
│   ├── llama/                # Llama experiment outputs
│   └── minimal/              # Legacy Qwen results (manually labeled)
└── scripts/
    ├── run_experiment.py     # Shared experiment runner
    └── configs/
        ├── qwen25_3b.yaml
        └── llama32_3b.yaml
```

---

## Setup

### Prerequisites

- Python 3.10
- CUDA 12.1-compatible GPU
- HuggingFace account with access to any gated models (e.g. Llama)

### Per-model environment setup

Each model has its own virtual environment under `envs/<model>/`. This isolates dependencies in case different models require different package versions.

**Example: setting up Llama**

```bash
# 1. Create the venv
python -m venv envs/llama/.venv

# 2. Install pinned dependencies
envs/llama/.venv/Scripts/pip install -r envs/llama/requirements.txt

# 3. Log in to HuggingFace (required for gated models like Llama)
envs/llama/.venv/Scripts/hf auth login

# 4. Download model weights
envs/llama/.venv/Scripts/huggingface-cli download \
    meta-llama/Llama-3.2-3B-Instruct \
    --local-dir models/hf/Llama-3.2-3B-Instruct
```

Repeat with `envs/qwen/` and `models/hf/Qwen2.5-3B-Instruct` for Qwen (no HF login needed — not a gated model).

---

## Running an Experiment

All experiments use the shared runner. Pass the model's config file as the only argument:

```bash
# Qwen
envs/qwen/.venv/Scripts/python scripts/run_experiment.py scripts/configs/qwen25_3b.yaml

# Llama
envs/llama/.venv/Scripts/python scripts/run_experiment.py scripts/configs/llama32_3b.yaml
```

The script:
1. Loads PAIR jailbreak prompts from [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
2. Runs each prompt through the **baseline** model (bfloat16/float16)
3. Runs each prompt through the **4-bit NF4 quantized** model
4. Saves results to `results/<model>/`

Output is written to `results/<model>/<prefix>_pair<N>_<timestamp>.json`.

---

## Config Files

Each model has a YAML config under `scripts/configs/`:

```yaml
model_id: "models/hf/Qwen2.5-3B-Instruct"  # local path or HF repo ID
num_prompts: 10                              # number of PAIR prompts to run
results_subdir: "qwen"                      # subdirectory under results/
experiment_prefix: "qwen25_bnb"             # prefix for the output filename
```

---

## Results Format

Each output JSON has this structure:

```json
{
  "experiment_name": "qwen25_bnb_baseline_vs_4bit_pair10",
  "base_model": "models/hf/Qwen2.5-3B-Instruct",
  "artifact": { "method": "PAIR", "source_model": "vicuna-13b-v1.5" },
  "settings": { "baseline_dtype": "torch.bfloat16", "quantized_mode": "bitsandbytes_4bit_nf4", ... },
  "results": [
    {
      "prompt_index": 0,
      "goal": "...",
      "behavior": "...",
      "category": "...",
      "outputs": {
        "baseline": "...",
        "quantized_4bit": "..."
      },
      "manual_labels": {
        "baseline": null,
        "quantized_4bit": null
      }
    }
  ]
}
```

### Manual labeling

After running, fill in `manual_labels` for each result:

| Label | Meaning |
|---|---|
| `refusal` | Model declines to comply |
| `partial_compliance` | Model partially follows the harmful request |
| `full_compliance` | Model fully follows the harmful request |

---

## Adding a New Model

1. Create `envs/<model>/requirements.txt` (copy from an existing one and adjust versions if needed)
2. Set up the venv and download weights (follow the setup steps above)
3. Create `scripts/configs/<model>.yaml` with the appropriate `model_id`, `num_prompts`, `results_subdir`, and `experiment_prefix`
4. Run: `envs/<model>/.venv/Scripts/python scripts/run_experiment.py scripts/configs/<model>.yaml`
