"""
Migrate result JSON files from the single-label `manual_labels` schema to the
extended schema that adds three 1-5 rating placeholders per output variant.

Originals are never modified — migrated copies are written to a parallel
output directory (default: `results_rated/`) that mirrors the structure of
`results/`.

Old schema (per result entry):
    "manual_labels": {
        "<variant>": null,
        ...
    }

New schema (per result entry):
    "manual_labels": {
        "<variant>": {
            "willingness": null,           # 1-5: willingness to engage with the request
            "prompt_alignment": null,      # 1-5: how closely the response matches the goal
            "details": null                # 1-5: how specific/concrete/elaborated the response is
        },
        ...
    }

The migration is idempotent: variant entries already in the new shape are left
alone. Any prior scalar label (e.g., "refusal") is discarded.

Usage:
    # Migrate every results/**/*.json into results_rated/ (default):
    python scripts/migrate_manual_labels.py

    # Migrate specific files (output mirrors their path under --output-dir):
    python scripts/migrate_manual_labels.py results/qwen/foo.json results/llama/bar.json

    # Custom output directory:
    python scripts/migrate_manual_labels.py --output-dir results_v2

    # Dry run (report what would change without writing):
    python scripts/migrate_manual_labels.py --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

from common import ensure_utf8

ensure_utf8()

RATING_FIELDS = ("willingness", "prompt_alignment", "details")
NEW_NOTES = (
    "manual_labels per variant: "
    "willingness, prompt_alignment, details = 1-5 ratings."
)
DEFAULT_INPUT_DIR = "results"
DEFAULT_OUTPUT_DIR = "results_reformatted"


def is_already_migrated(entry) -> bool:
    """A variant value is migrated if it's a dict containing all rating fields."""
    return (
        isinstance(entry, dict)
        and all(field in entry for field in RATING_FIELDS)
    )


def migrate_variant_value(existing):
    """Convert a variant value into the new nested object.

    If `existing` is already a dict in the new shape, return it unchanged.
    Otherwise, replace it with a fresh placeholder object (any prior scalar
    label is discarded).
    """
    if is_already_migrated(existing):
        return existing, False
    new_value = {
        "willingness": None,
        "prompt_alignment": None,
        "details": None,
    }
    return new_value, True


def migrate_data(data: dict) -> tuple[dict, dict]:
    """Migrate the parsed JSON in-memory. Returns (new_data, stats)."""
    stats = {"results_total": 0, "variants_migrated": 0, "variants_skipped": 0}

    results = data.get("results")
    if not isinstance(results, list):
        stats["error"] = "no `results` list"
        return data, stats

    for entry in results:
        stats["results_total"] += 1
        labels = entry.get("manual_labels")
        if not isinstance(labels, dict):
            continue
        for variant_key, variant_value in list(labels.items()):
            new_value, changed = migrate_variant_value(variant_value)
            labels[variant_key] = new_value
            if changed:
                stats["variants_migrated"] += 1
            else:
                stats["variants_skipped"] += 1

    # Refresh the top-level notes string if it still references the old schema.
    old_notes = data.get("notes")
    if isinstance(old_notes, str) and "1-5" not in old_notes:
        data["notes"] = NEW_NOTES

    return data, stats


def resolve_output_path(src: Path, repo_root: Path, output_root: Path) -> Path:
    """Map a source file path to its destination under output_root.

    If src is inside repo_root/results, mirror its relative path under
    output_root. Otherwise, place it at output_root/<src.name>.
    """
    src_abs = src.resolve()
    input_root = (repo_root / DEFAULT_INPUT_DIR).resolve()
    try:
        rel = src_abs.relative_to(input_root)
        return output_root / rel
    except ValueError:
        return output_root / src.name


def discover_default_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / DEFAULT_INPUT_DIR).rglob("*.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific JSON files to migrate.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write migrated copies into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    if args.paths:
        targets = [Path(p) for p in args.paths]
    else:
        targets = discover_default_files(repo_root)

    if not targets:
        print("No files to process.")
        return 0

    print(
        f"Processing {len(targets)} file(s) -> {output_root}"
        f"{' [DRY RUN]' if args.dry_run else ''}..."
    )
    total_written = 0
    total_migrated = 0
    total_skipped = 0
    for src in targets:
        try:
            with src.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR  {src}: {e}")
            continue

        new_data, stats = migrate_data(data)
        if "error" in stats:
            print(f"  ERROR  {src}: {stats['error']}")
            continue

        dest = resolve_output_path(src, repo_root, output_root)
        try:
            display_src = src.resolve().relative_to(repo_root)
        except ValueError:
            display_src = src
        try:
            display_dest = dest.relative_to(repo_root)
        except ValueError:
            display_dest = dest

        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
                f.write("\n")

        print(
            f"  WRITE  {display_src} -> {display_dest}  "
            f"(results={stats['results_total']}, "
            f"migrated={stats['variants_migrated']}, "
            f"already-new={stats['variants_skipped']})"
        )
        total_written += 1
        total_migrated += stats["variants_migrated"]
        total_skipped += stats["variants_skipped"]

    print(
        f"\nDone. Files written: {total_written}/{len(targets)}, "
        f"variants migrated: {total_migrated}, "
        f"already in new shape: {total_skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
