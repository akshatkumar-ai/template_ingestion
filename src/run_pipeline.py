#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Orchestrator that runs the template-ingestion pipeline stages in order:

  1. Template Extract      (src/template_extract/main.py)
  2. Batch Transform       (src/instruction_transform/batch_transform.py)
  3. Section Dependencies  (src/section_dependency/section_dependencies.py)

Usage
-----
    # Run all stages:
    python src/run_pipeline.py --from 1 --until 3

    # Run only Template Extract:
    python src/run_pipeline.py --only 1

    # Skip Template Extract, run stages 2 and 3:
    python src/run_pipeline.py --from 2 --until 3

    # Run just Batch Transform:
    python src/run_pipeline.py --only 2

    # Interactive mode — prompts you to pick:
    python src/run_pipeline.py

Any extra arguments after '--' are forwarded to every stage script.
"""

import subprocess
import sys
import os
import argparse

# ── Pipeline definition ───────────────────────────────────────────────────────

# Each stage is (label, module path relative to repo root)
STAGES = [
    ("Template Extract",     "src/template_extract/main.py"),
    ("Batch Transform",      "src/instruction_transform/batch_transform.py"),
    ("Section Dependencies", "src/section_dependency/section_dependencies.py"),
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_stages(highlight_stages=None):
    """Print the available pipeline stages, optionally highlighting a list of stages."""
    print("\n  Pipeline stages:")
    print("  " + "─" * 44)
    for i, (label, _) in enumerate(STAGES, start=1):
        marker = "  "
        if highlight_stages and i in highlight_stages:
            marker = "▶ "
        print(f"    {marker}{i}. {label}")
    print("  " + "─" * 44)
    print()


def run_stage(index: int, extra_args: list):
    """Run a single pipeline stage by index (0-based)."""
    label, script = STAGES[index]
    script_path = os.path.join(REPO_ROOT, script)

    if not os.path.isfile(script_path):
        print(f"  ✗ Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    separator = "═" * 60
    print(f"\n{separator}")
    print(f"  STAGE {index + 1}/{len(STAGES)} — {label}")
    print(f"  Script: {script}")
    print(separator)

    cmd = [sys.executable, script_path] + extra_args
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"\n  ✗ Stage {index + 1} ({label}) failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  ✓ Stage {index + 1} ({label}) completed successfully.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the template-ingestion pipeline (or a subset of stages).",
        epilog=(
            "Examples:\n"
            "  python src/run_pipeline.py --from 1 --until 3   # run all\n"
            "  python src/run_pipeline.py --from 2 --until 3   # skip stage 1\n"
            "  python src/run_pipeline.py --only 2             # just stage 2\n"
            "  python src/run_pipeline.py                      # interactive\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    valid = range(1, len(STAGES) + 1)
    parser.add_argument(
        "--from",
        type=int,
        choices=valid,
        default=None,
        dest="stage_from",
        metavar="N",
        help=f"Start from stage N (inclusive). Valid: 1–{len(STAGES)}.",
    )
    parser.add_argument(
        "--until",
        type=int,
        choices=valid,
        default=None,
        metavar="N",
        help=f"Run up to stage N (inclusive). Valid: 1–{len(STAGES)}.",
    )
    parser.add_argument(
        "--only",
        type=int,
        choices=valid,
        default=None,
        metavar="N",
        help=f"Run only stage N (shorthand for --from N --until N).",
    )
    # Everything after '--' is forwarded to each stage script
    args, extra = parser.parse_known_args()
    return args, extra


def _ask_stages(prompt_text: str) -> list:
    """Ask the user for comma-separated stage numbers."""
    while True:
        try:
            value = input(prompt_text).strip()
            if not value:
                continue
            parts = [p.strip() for p in value.split(",")]
            numbers = []
            for p in parts:
                n = int(p)
                if 1 <= n <= len(STAGES):
                    numbers.append(n)
                else:
                    print(f"  Stage number {n} is out of range.")
                    break
            else:
                return sorted(list(set(numbers)))
        except ValueError:
            print(f"  Please enter comma-separated numbers between 1 and {len(STAGES)}.")
        except (KeyboardInterrupt, EOFError):
            print("\n  Aborted.")
            sys.exit(0)


def prompt_user():
    """Interactively ask the user which stages to run."""
    print_stages()
    stages_to_run = _ask_stages(f"  Enter modules to run (comma-separated, 1-{len(STAGES)}): ")
    
    batch_start = None
    batch_end = None
    
    if 2 in stages_to_run:
        print("\n  [Batch Transform Options]")
        start_val = input("  Enter start row (leave blank for default 1): ").strip()
        if start_val.isdigit():
            batch_start = int(start_val)
        
        end_val = input("  Enter end row (leave blank for all): ").strip()
        if end_val.isdigit():
            batch_end = int(end_val)
            
    return stages_to_run, batch_start, batch_end


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args, extra = parse_args()

    stages_to_run = []
    batch_start = None
    batch_end = None

    # Resolve --only shorthand
    if args.only is not None:
        if args.stage_from is not None or args.until is not None:
            print("  ✗ --only cannot be combined with --from / --until.", file=sys.stderr)
            sys.exit(1)
        stages_to_run = [args.only]
    elif args.stage_from is not None or args.until is not None:
        stage_from = args.stage_from or 1
        stage_until = args.until or len(STAGES)
        if stage_until < stage_from:
            print(f"  ✗ --until ({stage_until}) cannot be less than --from ({stage_from}).", file=sys.stderr)
            sys.exit(1)
        stages_to_run = list(range(stage_from, stage_until + 1))
    else:
        # Interactive mode if neither flag was given
        stages_to_run, batch_start, batch_end = prompt_user()

    if not stages_to_run:
        print("  ✗ No stages selected.", file=sys.stderr)
        sys.exit(1)

    # Show what will run
    print_stages(highlight_stages=stages_to_run)
    print(f"  ▶ Running stages: {', '.join(map(str, stages_to_run))}\n")

    for i in stages_to_run:
        stage_index = i - 1
        stage_extra = list(extra)
        
        # If it's stage 2 and we have interactive start/end, append them
        if i == 2:
            if batch_start is not None:
                stage_extra.extend(["--start", str(batch_start)])
            if batch_end is not None:
                stage_extra.extend(["--end", str(batch_end)])

        run_stage(stage_index, stage_extra)

    print("\n" + "═" * 60)
    print(f"  ✓ Pipeline finished — {len(stages_to_run)} stage(s) completed.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
