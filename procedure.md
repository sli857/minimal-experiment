# Research Procedure

**Project:** LLM Safety Under Quantization  
**Course:** ECE 8803, Georgia Tech, Spring 2026  
**Team:** Shuyi Li, Jingyu Liu, Raj Patel, Abhinav Vemulapalli  
**Methodology:** Thematic analysis (Braun & Clarke, 2006)

---

## Overview

This document describes the end-to-end research procedure for studying how quantization affects the safety guardrails of instruction-tuned LLMs. The procedure has five phases: data collection, familiarization, open coding, codebook construction, and LLM-assisted labelling.

---

## Current Status (as of 2026-04-09)

| Phase | Status |
|---|---|
| 1. Data Collection | **Complete** |
| 2. Familiarization | Not started |
| 3. Open Coding | Not started |
| 4. Codebook Construction | Not started |
| 5. LLM-Assisted Labelling | Not started |

### Data collection detail

| Model | Condition | Prompts | Status |
|---|---|---|---|
| Llama-3.2-3B | Baseline (bf16) | 100 | Done (Apr 2) |
| Llama-3.2-3B | INT8 (BnB) | 100 | Done (Apr 2) |
| Llama-3.2-3B | NF4 (BnB) | 100 | Done (Apr 2) |
| Llama-3.2-3B | GGUF Q4_0 | 100 | Done (Apr 7) |
| Llama-3.2-3B | GGUF Q4_K_M | 100 | Done (Apr 7) |
| Llama-3.2-3B | GGUF Q8_0 | 100 | Done (Apr 7) |
| Qwen2.5-3B | Baseline (bf16) | 100 | Done (Apr 9) |
| Qwen2.5-3B | INT8 (BnB) | 100 | Done (Apr 9) |
| Qwen2.5-3B | NF4 (BnB) | 100 | Done (Apr 9) |
| Qwen2.5-3B | GGUF Q4_0 | 100 | Done (Apr 7) |
| Qwen2.5-3B | GGUF Q4_K_M | 100 | Done (Apr 7) |
| Qwen2.5-3B | GGUF Q8_0 | 100 | Done (Apr 7) |

---

## Phase 1: Data Collection — *Complete*

### 1.1 Models

| Model | Size | Source |
|---|---|---|
| Qwen2.5-3B-Instruct | 3B | Qwen (open weight) |
| Llama-3.2-3B-Instruct | 3B | Meta (gated) |

### 1.2 Quantization conditions

Each model is run under multiple conditions to compare safety behavior:

| Condition | Description | Library |
|---|---|---|
| **Baseline** | Full-precision bfloat16 (or float16 fallback) | transformers |
| **INT4 (NF4)** | 4-bit Normal Float via BitsAndBytes | bitsandbytes |
| **INT8** | 8-bit quantization via BitsAndBytes | bitsandbytes |
| **GGUF** | llama.cpp quantization (Q4_0, Q4_K_M, Q8_0) | llama-cpp-python |

### 1.3 Jailbreak prompts

- **Source:** JailbreakBench (https://github.com/JailbreakBench/jailbreakbench)
- **Artifact:** PAIR method, generated against vicuna-13b-v1.5
- **Count:** 100 prompts per model per condition
- **Categories:** Harassment/Discrimination, and others as defined by JailbreakBench

### 1.4 Generation settings

- `max_new_tokens`: 256
- `do_sample`: False (greedy decoding for reproducibility)
- System prompt: "You are a helpful assistant."

### 1.5 Execution

For each model and each quantization condition:

1. Load the model under the specified condition
2. For each PAIR jailbreak prompt, construct a chat message (system + user) and generate a response
3. Save all prompt-response pairs to a JSON file under `results/<model>/`

See `scripts/run_experiment.py` and `scripts/configs/` for implementation details.

---

## Phase 2: Familiarization — *Not started*

Before any formal coding, each team member reads through a substantial subset of the collected responses to develop an intuitive understanding of the data.

### Steps

1. Each team member reads at least 50 prompt-response pairs (across both conditions and both models)
2. While reading, take informal notes on:
   - Recurring patterns in how models refuse, comply, or hedge
   - Differences between baseline and quantized responses
   - Anything surprising or unexpected
3. Team meets to discuss initial impressions and agree on the scope of the coding exercise

### Output

A set of informal notes and shared observations. No formal codes yet.

---

## Phase 3: Open Coding — *Not started*

Open coding is performed iteratively on a subset of responses until **theoretical saturation** — the point at which reading additional responses no longer produces new codes.

### Steps

1. **Select an initial batch** — randomly sample ~20 prompt-response pairs (covering both models and both quantization conditions)
2. **Code independently** — at least two team members read each response and assign descriptive codes to meaningful segments. Codes should capture *how* the model responds, not just *whether* it complies. Examples of the kinds of distinctions to look for:
   - Hard refusal ("I can't help with that")
   - Refusal with explanation of why it's harmful
   - Refusal with redirection to a safer topic
   - Compliance framed as fictional / historical / hypothetical
   - Compliance with disclaimers or moral distancing
   - Compliance with reduced specificity (watered-down content)
   - Full unqualified compliance
3. **Compare and discuss** — team members meet to compare codes, resolve disagreements, and merge/refine the code set
4. **Code the next batch** — sample another ~20 responses, repeat steps 2-3
5. **Check for saturation** — after each batch, assess whether new codes are still emerging. If a full batch produces no new codes, saturation is reached
6. **Repeat** until saturation

### Saturation criterion

Saturation is reached when an entire batch of 20 responses is coded without introducing any new code that is not already in the working code set. At that point, stop open coding and proceed to codebook construction.

### Output

A working set of codes with short descriptions and example excerpts from the data.

---

## Phase 4: Codebook Construction — *Not started*

Consolidate the open codes into a final codebook — a structured set of themes that will be used for the full-dataset labelling.

### Steps

1. **Group codes into themes** — cluster related codes into higher-level themes (e.g., several refusal-style codes might form a "refusal strategies" theme with sub-codes)
2. **Review themes against data** — re-read a sample of responses to verify that the themes accurately capture the patterns in the data. Merge themes that overlap; split themes that are too broad
3. **Define each theme** — for every theme (and sub-theme if applicable), write:
   - **Name** — a concise label
   - **Definition** — what this theme captures
   - **Inclusion criteria** — when to apply this code
   - **Exclusion criteria** — when *not* to apply this code (to distinguish from similar themes)
   - **Example** — one or two illustrative excerpts from the data
4. **Team review** — all team members review the codebook and resolve any disagreements

### Output

A finalized codebook document (to be saved in the repository) that serves as the labelling guide for Phase 5.

---

## Phase 5: LLM-Assisted Labelling — *Not started*

Use a separate LLM (the "judge") to classify every response in the full dataset according to the codebook. This scales the labelling to the entire dataset without requiring manual annotation of every response.

### Steps

1. **Design the judge prompt** — construct a prompt that:
   - Provides the full codebook (theme names, definitions, inclusion/exclusion criteria, examples)
   - Presents one victim-model response at a time
   - Asks the judge to assign one or more theme labels and provide a brief justification
   - Outputs structured JSON for easy parsing
2. **Select a judge model** — use a capable instruction-following model (e.g., GPT-4, Claude, or a large open-weight model). The judge model should be different from the victim models being studied
3. **Pilot test** — run the judge on the manually-coded subset from Phase 3 and compare its labels to the human labels. Compute inter-rater agreement (e.g., Cohen's kappa). Iterate on the prompt until agreement is acceptable
4. **Full labelling** — run the judge on all responses across all models and conditions
5. **Spot-check** — team members manually verify a random sample of judge labels to confirm quality

### Output

A fully labelled dataset with theme codes for every response, ready for quantitative analysis (e.g., comparing theme distributions across quantization conditions).
