"""
batch_transform.py
------------------
Reads the system prompt from a .txt file and processes entries n1 to n2
from a JSON file that follows the nested section-instructions schema.

The JSON file must have the structure::

    {
      "sections": [
        {
          "section_number": "1",
          "section_name": "INTRODUCTION",
          "instructions": "...",
          "subsections": [ ... ]
        },
        ...
      ]
    }

The script flattens all sections and subsections recursively, treating each
node as one "row" to process.  The three fields used for instruction
generation are:

    JSON key          Role (equivalent CSV column)
    ──────────────    ─────────────────────────────
    section_number    section_number
    section_name      section_title
    instructions      template_instructions

After transformation the result is written back into the same JSON node as:
    transformed_output   – the Claude-generated instruction
    reasoning            – the model's reasoning

Also reads TOC.md and includes it as context in each prompt.
Caches the final prompt and JSON response in outputs/<section_number>/.

Usage
-----
    python batch_transform.py \\
        --system_prompt system_prompt.txt \\
        --json section_instructions.json \\
        [--start 1] [--end 10] \\
        [--section 1.3.1] \\
        [--list] \\
        [--authoring_type protocol|csr|sap] \\
        [--model_id <bedrock-model-arn>]

Arguments
---------
--system_prompt       Path to the .txt file containing the full system prompt.
--json                Path to the JSON file containing section instructions.
                      The file will be updated in place with new
                      'transformed_output' and 'reasoning' fields.
--start               (Optional) Starting flat index (1-based). Defaults to 1.
                      Use --list to discover the flat index of any section.
--end                 (Optional) Ending flat index (1-based). Defaults to all.
--section             (Optional) Process exactly ONE section identified by its
                      section_number string (e.g. --section 1.3.1).
                      Mutually exclusive with --start / --end.
--list                (Optional) Print the full flat index of all sections
                      (number, index, name) and exit without calling Bedrock.
                      Use this to find the flat index you need for --start/--end.
--authoring_type      (Optional) Type of authoring: 'protocol', 'csr', or 'sap'.
                      Determines which source documents are used as anchor and
                      secondary sources. Defaults to 'protocol'.
--model_id            (Optional) AWS Bedrock model ARN / inference-profile ID.
                      Defaults to the Claude Sonnet 4 inference profile.
"""

import argparse
import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# AWS Bedrock helpers (ported from transform.py)
# ---------------------------------------------------------------------------

def get_bedrock_client():
    print("*" * 50)
    print("Connecting to AWS BEDROCK....")

    import boto3
    from dotenv import load_dotenv

    load_dotenv(".env")

    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
    AWS_REGION = "us-east-1"

    if not AWS_ACCESS_KEY_ID:
        raise ValueError("AWS_ACCESS_KEY_ID is missing in the environment variables.")
    if not AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS_SECRET_ACCESS_KEY is missing in the environment variables.")
    if not AWS_SESSION_TOKEN:
        raise ValueError("AWS_SESSION_TOKEN is missing in the environment variables.")

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
            config=boto3.session.Config(read_timeout=2000),
        )
        print("*" * 50)
        return client
    except Exception as e:
        print(e)
        raise ValueError("Error in loading BEDROCK client!")


def generate(prompt: str, model_id: str, client=None) -> str:
    from dotenv import load_dotenv

    load_dotenv(".env")

    if "claude" not in model_id.lower():
        raise ValueError(
            f"Unknown model_id '{model_id}'. Only Claude models are currently supported."
        )

    print(f"Generating response using {model_id}")

    if client is None:
        try:
            client = get_bedrock_client()
        except Exception:
            raise ValueError("Error in creating client session!!!")

    try:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt.strip()}],
                }
            ],
        }

        response = client.invoke_model(
            modelId=model_id, body=json.dumps(native_request)
        )
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]

    except Exception as e:
        print(e)
        raise ValueError("Error in generation!")


def generate_with_metrics(prompt: str, model_id: str, client) -> tuple[str, dict]:
    """
    Stream a prompt to Bedrock and return (response_text, metrics).

    Uses invoke_model_with_response_stream so large responses don't buffer
    silently and risk timing out.  Progress dots are printed as chunks arrive.

    Returns
    -------
    text    : the full model response string
    metrics : dict with keys input_tokens, output_tokens, stop_reason, wall_seconds
    """
    if "claude" not in model_id.lower():
        raise ValueError(
            f"Unknown model_id '{model_id}'. Only Claude models are supported."
        )

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt.strip()}],
            }
        ],
    }

    print(f"  Streaming via {model_id}")
    print("  Progress: ", end="", flush=True)
    t_start = time.perf_counter()

    try:
        response = client.invoke_model_with_response_stream(
            modelId=model_id, body=json.dumps(native_request)
        )
    except Exception as e:
        print(e)
        raise ValueError("Error starting streaming request!")

    chunks        = []
    input_tokens  = "N/A"
    output_tokens = "N/A"
    stop_reason   = "N/A"

    try:
        for event in response["body"]:
            chunk_data = json.loads(event["chunk"]["bytes"])
            chunk_type = chunk_data.get("type", "")

            if chunk_type == "content_block_delta":
                delta = chunk_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
                    print(".", end="", flush=True)

            elif chunk_type == "message_delta":
                usage         = chunk_data.get("usage", {})
                output_tokens = usage.get("output_tokens", "N/A")
                stop_reason   = chunk_data.get("delta", {}).get("stop_reason", stop_reason)

            elif chunk_type == "message_start":
                usage        = chunk_data.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", "N/A")

    except Exception as e:
        print(e)
        raise ValueError("Error reading streaming response!")

    t_end = time.perf_counter()
    print()  # newline after progress dots

    metrics = {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "stop_reason":   stop_reason,
        "wall_seconds":  round(t_end - t_start, 2),
    }
    return "".join(chunks), metrics


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Authoring mode configuration
# ---------------------------------------------------------------------------

AUTHORING_SOURCES = {
    "protocol": {
        "anchor": "Synopsis",
        "secondary": "Investigators Brochure, Industry Literature, Guidelines"
    },
    "csr": {
        "anchor": "Protocol and/or SAP and/or MOP",
        "secondary": "Protocol Amendments - Summary of Changes, Narratives & Forms for Death / SAE / Other Significant AEs, Industry Literature, Guidelines, Discussion references, eCRFs"
    },
    "sap": {
        "anchor": "Protocol",
        "secondary": "eCRFs"
    }
}


def build_prompt(
    system_prompt: str,
    section_context: str,
    toc_content: str,
    input_instruction: str,
    authoring_type: str = "protocol",
) -> str:
    """
    Wraps the raw input_instruction inside the <input_instruction> XML tag
    and appends it to the system prompt, matching the notebook structure.
    Also includes the current section context and the TOC as context.
    Replaces authoring mode placeholders based on authoring_type.
    """
    # Get authoring sources for the given type
    sources = AUTHORING_SOURCES.get(authoring_type.lower(), AUTHORING_SOURCES["protocol"])

    # Replace placeholders in system prompt
    prompt_with_authoring = system_prompt.replace(
        "<AUTHORING_TYPE>", authoring_type
    ).replace(
        "<ANCHOR_SOURCE_TYPE>", sources["anchor"]
    ).replace(
        "<LIST_OF_SUPPORTING_SOURCE_TYPES>", sources["secondary"]
    )

    return (
        prompt_with_authoring.rstrip()
        + "\n\n<section_context>\n"
        + section_context.strip()
        + "\n</section_context>\n\n<toc_context>\n"
        + toc_content.strip()
        + "\n</toc_context>\n\n<input_instruction>\n"
        + input_instruction.strip()
        + "\n</input_instruction>\n"
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

# Sentinel values used to detect entries that should be skipped.
_SKIP_INSTRUCTIONS = {"no instructions found", ""}


def _is_skippable(instructions: str) -> bool:
    """Return True when instructions carry no actionable content."""
    return instructions.strip().lower() in _SKIP_INSTRUCTIONS


def flatten_sections(node_list: list) -> list:
    """
    Recursively flatten a nested list of section nodes into a flat list.

    Each element of the returned list is a *reference* to the original dict
    in the loaded JSON object, so mutations (adding keys) propagate back.

    Required keys per node: section_number, section_name, instructions.
    Optional key           : subsections (list of child nodes).
    """
    flat = []
    for node in node_list:
        flat.append(node)
        children = node.get("subsections", [])
        if children:
            flat.extend(flatten_sections(children))
    return flat


def load_json_file(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Failed to parse JSON file '{path}': {exc}", file=sys.stderr)
            sys.exit(1)


def save_json_file(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = (
    "arn:aws:bedrock:us-east-1:533267065792:"
    "inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
)
DEFAULT_SYSTEM_PROMPT = "system_prompt.txt"
DEFAULT_JSON = "section_instructions.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch transform section instructions from a JSON file using "
            "AWS Bedrock (Claude) and write results back into the same JSON."
        )
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            f"Path to the .txt file containing the system prompt. "
            f"Defaults to: {DEFAULT_SYSTEM_PROMPT}"
        ),
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON,
        help=(
            f"Path to the JSON file containing section instructions. "
            f"The file will be updated in place with 'transformed_output' and "
            f"'reasoning' fields. Defaults to: {DEFAULT_JSON}"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "Starting flat index (1-based) to process. Defaults to 1. "
            "Run --list first to see all flat indices."
        ),
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending flat index (1-based) to process. Defaults to all entries.",
    )
    parser.add_argument(
        "--section",
        default=None,
        metavar="SECTION_NUMBER",
        help=(
            "Process exactly one section by its section_number string "
            "(e.g. --section 1.3.1). Mutually exclusive with --start / --end."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "Print the flat index of every section (index | section_number | "
            "section_name) and exit. Use this to find what to pass to "
            "--start / --end or --section."
        ),
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=(
            f"AWS Bedrock model ARN or inference-profile ID. "
            f"Defaults to: {DEFAULT_MODEL_ID}"
        ),
    )
    parser.add_argument(
        "--authoring_type",
        default="protocol",
        choices=["protocol", "csr", "sap"],
        help="Authoring type: 'protocol', 'csr', or 'sap'. Defaults to 'protocol'.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help=(
            "Enable metric tracking. Uses streaming API to report per-section "
            "wall-clock time, input/output token counts, and stop_reason."
        ),
    )

    args = parser.parse_args()

    # Validate: --section is mutually exclusive with --start / --end
    if args.section and (args.start != 1 or args.end is not None):
        parser.error("--section cannot be combined with --start or --end.")

    return args


def read_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_flat_index(all_nodes: list) -> None:
    """Print a human-readable table of all flattened section nodes."""
    col_w = max(len(str(n.get("section_number", ""))) for n in all_nodes) + 2
    header = f"{'Index':>6}  {'Section Number':<{col_w}}  Section Name"
    print(header)
    print("-" * min(len(header) + 40, 100))
    for i, node in enumerate(all_nodes, start=1):
        num  = str(node.get("section_number", ""))
        name = node.get("section_name", "")
        skip = " [skip — no instructions]" if _is_skippable(node.get("instructions", "")) else ""
        print(f"{i:>6}  {num:<{col_w}}  {name}{skip}")


def main():
    args = parse_args()

    # ── Load JSON and flatten (needed for both --list and processing) ──────
    data = load_json_file(args.json)

    top_level_sections = data.get("sections")
    if not isinstance(top_level_sections, list):
        print(
            "ERROR: JSON root must contain a 'sections' array.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_nodes = flatten_sections(top_level_sections)
    total = len(all_nodes)

    # ── --list: print flat index and exit (no Bedrock call) ───────────────
    if args.list:
        print(f"Flat index for: {args.json}  ({total} total nodes)\n")
        print_flat_index(all_nodes)
        print(
            "\nTip: pass --section <section_number> to process one section, "
            "or --start <n> --end <m> for a range."
        )
        sys.exit(0)

    system_prompt = read_file(args.system_prompt, "System prompt")
    toc_content = read_file("TOC.md", "TOC")

    os.makedirs("outputs", exist_ok=True)

    # ── Determine nodes to process ─────────────────────────────────────────
    if args.section:
        # --section: find the node whose section_number matches exactly
        target = args.section.strip()
        nodes_to_process = [
            n for n in all_nodes
            if str(n.get("section_number", "")).strip() == target
        ]
        if not nodes_to_process:
            print(
                f"ERROR: No section found with section_number '{target}'.\n"
                f"Run with --list to see all available section numbers.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Processing section '{target}' (1 node).")
    else:
        # --start / --end: flat positional range
        start = max(1, args.start)
        end   = args.end
        if end is not None and end < start:
            print("ERROR: --end must be >= --start", file=sys.stderr)
            sys.exit(1)
        start_idx = start - 1          # convert to 0-based
        end_idx   = min(end, total) if end is not None else total
        nodes_to_process = all_nodes[start_idx:end_idx]
        print(f"Total flattened section nodes: {total}")
        print(f"Processing flat indices {start} to {end_idx} (1-based).")

    client = get_bedrock_client()

    # ── Process each node ──────────────────────────────────────────────────
    for node in nodes_to_process:
        # Map JSON keys → roles previously filled by CSV columns
        section_number: str = str(node.get("section_number", "")).strip()
        section_title: str  = node.get("section_name", "").strip()     # was section_title
        instructions: str   = node.get("instructions", "").strip()     # was template_instructions

        if _is_skippable(instructions):
            print(f"  Skipping section {section_number!r} (no actionable instructions).")
            node["transformed_output"] = ""
            node["reasoning"] = ""
            continue

        print(f"\nProcessing section {section_number!r}: {section_title}")

        section_context = (
            f"section_number: {section_number}\n"
            f"section_title: {section_title}"
        )

        prompt = build_prompt(
            system_prompt,
            section_context,
            toc_content,
            instructions,
            args.authoring_type,
        )

        # ── Call the LLM (with or without metrics) ─────────────────────────
        if args.metrics:
            result, metrics = generate_with_metrics(prompt, args.model_id, client)
            print(f"  Time: {metrics['wall_seconds']}s | "
                  f"in={metrics['input_tokens']} out={metrics['output_tokens']} tokens | "
                  f"stop={metrics['stop_reason']}")
            if metrics["stop_reason"] == "max_tokens":
                print(f"  WARNING: output for section {section_number!r} was truncated!")
        else:
            result = generate(prompt, args.model_id, client)

        # Cache prompt and raw response
        safe_number = section_number.replace("/", "_").replace("\\", "_")
        section_folder = f"outputs/{safe_number}"
        os.makedirs(section_folder, exist_ok=True)
        with open(f"{section_folder}/prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(f"{section_folder}/response.json", "w", encoding="utf-8") as f:
            f.write(result)

        # Parse the model response
        try:
            # Strip markdown code block wrapper if present
            clean_result = result.strip()
            if clean_result.startswith("```json"):
                clean_result = clean_result[7:]  # Remove ```json
            if clean_result.startswith("```"):
                clean_result = clean_result[3:]  # Remove ```
            if clean_result.endswith("```"):
                clean_result = clean_result[:-3]  # Remove trailing ```
            clean_result = clean_result.strip()

            parsed = json.loads(clean_result)
            node["transformed_output"] = parsed.get("transformed_instruction", "")
            node["reasoning"] = parsed.get("reasoning", "")
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parse error for section {section_number!r}: {e}")
            node["transformed_output"] = result  # Fallback to raw text
            node["reasoning"] = ""

        print(f"  Transformed section {section_number!r}.")

    # ── Write results back to the same JSON file ───────────────────────────
    save_json_file(data, args.json)
    print(f"\nUpdated JSON: {args.json}")


if __name__ == "__main__":
    main()