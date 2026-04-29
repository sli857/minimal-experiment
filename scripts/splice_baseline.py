import json
import shutil
import sys

def splice_baseline(target_file, source_file):
    # Create backup of target file
    backup_file = target_file + ".bak"
    shutil.copy2(target_file, backup_file)
    print(f"Created backup of target file at {backup_file}")

    with open(source_file, "r") as f:
        source_data = json.load(f)
    
    with open(target_file, "r") as f:
        target_data = json.load(f)

    # Map baseline outputs by prompt_index
    baseline_map = {}
    for res in source_data["results"]:
        baseline_map[res["prompt_index"]] = res["outputs"]["baseline"]

    # Append baseline to target results
    for res in target_data["results"]:
        pid = res["prompt_index"]
        if pid in baseline_map:
            # Reorder outputs dict so baseline is first
            old_outputs = res["outputs"]
            new_outputs = {"baseline": baseline_map[pid]}
            new_outputs.update(old_outputs)
            res["outputs"] = new_outputs

            old_labels = res.get("manual_labels", {})
            new_labels = {
                "baseline": {
                    "willingness": None,
                    "prompt_alignment": None,
                    "details": None
                }
            }
            new_labels.update(old_labels)
            res["manual_labels"] = new_labels

    # Copy the baseline dtype info if missing
    if "settings" in source_data and "baseline_dtype" in source_data["settings"]:
        target_data["settings"]["baseline_dtype"] = source_data["settings"]["baseline_dtype"]
    
    # Save the modified target data back
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(target_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully spliced baseline into {target_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python splice_baseline.py <target_file> <source_file>")
        sys.exit(1)
    
    splice_baseline(sys.argv[1], sys.argv[2])
