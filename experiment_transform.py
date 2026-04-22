"""
experiment_transform.py
-----------------------
EXPERIMENT: sends the ENTIRE flattened section list to the LLM in a single
call instead of iterating section by section (as batch_transform.py does).

Purpose
-------
Test whether Claude can process all sections in one shot and whether it stays
within the model's context / output token limits.  Wall-clock time and token
usage are printed after every run so the results are easy to compare.

How it works
------------
1. Load the JSON file and flatten all sections recursively.
2. Filter out sections with no actionable instructions.
3. Build ONE prompt that contains:
     - the system prompt (with authoring-mode placeholders resolved)
     - the TOC context
     - a JSON array of all sections: [{section_number, section_name, instructions}, ...]
4. Send the single prompt to Bedrock and time the call.
5. Expect the model to return a JSON array:
     [{section_number, transformed_instruction, reasoning}, ...]
6. Match each result back to the original node by section_number and write
   transformed_output + reasoning into it.
7. Save the updated hierarchical JSON back to disk.
8. Print a timing + token-usage summary.

Usage
-----
    python experiment_transform.py \\
        --system_prompt system_prompt2.txt \\
        --json section_instructions.json \\
        [--authoring_type protocol|csr|sap] \\
        [--model_id <bedrock-model-arn>] \\
        [--output experiment_output.json]

Arguments
---------
--system_prompt   Path to the .txt system prompt (bulk-mode version).
                  Defaults to: system_prompt2.txt
--json            Path to the JSON file with the nested section schema.
                  Defaults to: section_instructions.json
--authoring_type  'protocol', 'csr', or 'sap'.  Defaults to 'protocol'.
--model_id        AWS Bedrock model ARN.  Defaults to Claude Sonnet 4.
--output          Path for the result JSON file.
                  Defaults to: experiment_output.json
"""

import argparse
import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# AWS Bedrock helpers
# ---------------------------------------------------------------------------

def get_bedrock_client():
    print("*" * 50)
    print("Connecting to AWS BEDROCK....")

    import boto3
    from dotenv import load_dotenv

    load_dotenv(".env")

    AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN     = os.getenv("AWS_SESSION_TOKEN")
    AWS_REGION            = "us-east-1"

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


def generate_with_metrics(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    client,
) -> tuple[str, dict]:
    """
    Stream a request to Bedrock and return (response_text, usage_metrics).

    system_prompt  → passed via the dedicated top-level `system` field.
                     The model treats this as persistent role/persona context
                     that is always in scope, separate from the user turn.
    user_prompt    → the actual task content (TOC + sections array).

    Why separate system and user?
    - Claude is specifically trained to distinguish system-level instructions
      (rules, persona, output format) from user-level content (the data to
      act on). Mixing them in one message reduces that separation.
    - With the system field, the model never confuses the transformation rules
      with the section instructions it is supposed to transform.
    - This also sets up correct behaviour if prompt caching is enabled later:
      the system prompt (which never changes between runs) can be cached
      independently of the variable user turn.

    usage_metrics keys:
        input_tokens   – tokens consumed by the prompt
        output_tokens  – tokens in the model response
        stop_reason    – 'end_turn' = finished naturally, 'max_tokens' = truncated
        wall_seconds   – wall-clock time for the full streaming call
    """
    if "claude" not in model_id.lower():
        raise ValueError(
            f"Unknown model_id '{model_id}'. Only Claude models are supported."
        )

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64000,          # Claude Sonnet 4 ceiling
        "temperature": 0.1,
        "system": system_prompt,      # ← dedicated system field, NOT in messages
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt.strip()}],
            }
        ],
    }

    print(f"Streaming bulk request to {model_id} ...")
    print("Progress: ", end="", flush=True)
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
    stop_reason   = "N/A"   # "end_turn" = finished naturally, "max_tokens" = truncated

    try:
        for event in response["body"]:
            chunk_data = json.loads(event["chunk"]["bytes"])
            chunk_type = chunk_data.get("type", "")

            if chunk_type == "content_block_delta":
                delta = chunk_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
                    print(".", end="", flush=True)   # live progress

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

    text = "".join(chunks)
    metrics = {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "stop_reason":   stop_reason,
        "wall_seconds":  round(t_end - t_start, 2),
    }
    return text, metrics


# ---------------------------------------------------------------------------
# Authoring mode configuration
# ---------------------------------------------------------------------------

AUTHORING_SOURCES = {
    "protocol": {
        "anchor": "Synopsis",
        "secondary": "Investigators Brochure, Industry Literature, Guidelines",
    },
    "csr": {
        "anchor": "Protocol and/or SAP and/or MOP",
        "secondary": (
            "Protocol Amendments - Summary of Changes, Narratives & Forms for "
            "Death / SAE / Other Significant AEs, Industry Literature, Guidelines, "
            "Discussion references, eCRFs"
        ),
    },
    "sap": {
        "anchor": "Protocol",
        "secondary": "eCRFs",
    },
}


# ---------------------------------------------------------------------------
# Prompt assembly  –  BULK version
# ---------------------------------------------------------------------------

def build_bulk_prompt(
    toc_content: str,
    sections_payload: list[dict],
) -> str:
    """
    Build the USER-TURN content only: TOC context + the JSON array of sections.

    The system prompt (role, rules, authoring mode) is passed separately via
    the `system` field in the Bedrock request and is NOT included here.
    This keeps the user turn focused purely on the variable input data.

    sections_payload is a list of dicts:
        [{"section_number": "1.1", "section_name": "...", "instructions": "..."}, ...]
    """
    sections_json = json.dumps(sections_payload, indent=2, ensure_ascii=False)

    return (
        "<toc_context>\n"
        + toc_content.strip()
        + "\n</toc_context>\n\n<input_sections>\n"
        + sections_json
        + "\n</input_sections>\n"
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

_SKIP_INSTRUCTIONS = {"no instructions found", ""}


def _is_skippable(instructions: str) -> bool:
    return instructions.strip().lower() in _SKIP_INSTRUCTIONS


def flatten_sections(node_list: list) -> list:
    """Depth-first flatten; returned items are references into the original tree."""
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
            print(f"ERROR: Cannot parse JSON '{path}': {exc}", file=sys.stderr)
            sys.exit(1)


def save_json_file(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def read_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def strip_code_fence(text: str) -> str:
    """Remove optional ```json ... ``` wrapper from model output."""
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = (
    "arn:aws:bedrock:us-east-1:533267065792:"
    "inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
)
DEFAULT_SYSTEM_PROMPT = "system_prompt2.txt"
DEFAULT_JSON         = "section_instructions.json"
DEFAULT_OUTPUT       = "experiment_output.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "EXPERIMENT: transform ALL section instructions in a single LLM call "
            "and measure time + token usage."
        )
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=f"Path to the bulk-mode system prompt. Defaults to: {DEFAULT_SYSTEM_PROMPT}",
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON,
        help=f"Path to the section-instructions JSON. Defaults to: {DEFAULT_JSON}",
    )
    parser.add_argument(
        "--authoring_type",
        default="protocol",
        choices=["protocol", "csr", "sap"],
        help="Authoring type. Defaults to 'protocol'.",
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"Bedrock model ARN. Defaults to Claude Sonnet 4.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file path. Defaults to: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    system_prompt = read_file(args.system_prompt, "System prompt")
    toc_content   = read_file("TOC.md", "TOC")

    # ── Load + flatten JSON ────────────────────────────────────────────────
    data = load_json_file(args.json)
    top_sections = data.get("sections")
    if not isinstance(top_sections, list):
        print("ERROR: JSON root must contain a 'sections' array.", file=sys.stderr)
        sys.exit(1)

    all_nodes = flatten_sections(top_sections)
    total     = len(all_nodes)
    print(f"Total section nodes (flattened): {total}")

    # ── Build the payload that goes into the prompt ────────────────────────
    # Only include sections that have actionable instructions.
    # Sections with "No instructions found" are marked directly in the tree.
    actionable = []
    skipped    = []
    for node in all_nodes:
        instr = node.get("instructions", "").strip()
        if _is_skippable(instr):
            node["transformed_output"] = ""
            skipped.append(node.get("section_number", "?"))
        else:
            actionable.append({
                "section_number": str(node.get("section_number", "")),
                "section_name":   node.get("section_name", ""),
                "instructions":   instr,
            })

    print(f"  Actionable sections to send:  {len(actionable)}")
    print(f"  Sections skipped (no instr.): {len(skipped)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    # ── Resolve authoring placeholders and split system / user turns ──────
    os.makedirs("outputs", exist_ok=True)
    sources = AUTHORING_SOURCES.get(args.authoring_type.lower(), AUTHORING_SOURCES["protocol"])
    resolved_system_prompt = (
        system_prompt
        .replace("<AUTHORING_TYPE>", args.authoring_type)
        .replace("<ANCHOR_SOURCE_TYPE>", sources["anchor"])
        .replace("<LIST_OF_SUPPORTING_SOURCE_TYPES>", sources["secondary"])
    )

    # User turn: only the variable data (TOC + section array)
    user_prompt = build_bulk_prompt(toc_content, actionable)

    with open("outputs/experiment_prompt.txt", "w", encoding="utf-8") as fh:
        fh.write("=== SYSTEM PROMPT (sent via `system` field) ===\n")
        fh.write(resolved_system_prompt)
        fh.write("\n\n=== USER TURN (sent via `messages[0]`) ===\n")
        fh.write(user_prompt)
    print(f"\nPrompt cached → outputs/experiment_prompt.txt")
    print(f"  System prompt : {len(resolved_system_prompt):,} chars")
    print(f"  User turn     : {len(user_prompt):,} chars\n")

    # ── Single LLM call (streaming, system/user split) ─────────────────────
    client = get_bedrock_client()
    raw_response, metrics = generate_with_metrics(
        resolved_system_prompt, user_prompt, args.model_id, client
    )

    # Cache raw response
    with open("outputs/experiment_response_raw.txt", "w", encoding="utf-8") as fh:
        fh.write(raw_response)

    # ── Print timing + token summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPERIMENT METRICS")
    print("=" * 60)
    print(f"  Wall-clock time : {metrics['wall_seconds']} seconds")
    print(f"  Input tokens    : {metrics['input_tokens']}")
    print(f"  Output tokens   : {metrics['output_tokens']}")
    print(f"  Stop reason     : {metrics['stop_reason']}")
    if metrics["stop_reason"] == "max_tokens":
        print("  WARNING: Output was TRUNCATED - model hit the max_tokens limit.")
        print("           The JSON array may be incomplete. Consider splitting into batches.")
    elif metrics["stop_reason"] == "end_turn":
        print("  OK: Model finished naturally - output is complete.")
    print("=" * 60 + "\n")

    # ── Parse the bulk response ────────────────────────────────────────────
    # Build a lookup from section_number → original node for fast matching.
    node_index = {
        str(n.get("section_number", "")).strip(): n
        for n in all_nodes
    }

    try:
        results = json.loads(strip_code_fence(raw_response))
        if not isinstance(results, list):
            raise ValueError("Model response is not a JSON array.")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: Could not parse model response as JSON array: {exc}")
        print("Raw response saved to outputs/experiment_response_raw.txt")
        sys.exit(1)

    # ── Match results back into the tree ───────────────────────────────────
    matched   = 0
    unmatched = []

    for item in results:
        sec_num = str(item.get("section_number", "")).strip()
        node    = node_index.get(sec_num)
        if node is None:
            unmatched.append(sec_num)
            continue
        node["transformed_output"] = item.get("transformed_instruction", "")
        matched += 1

    print(f"Results matched back into tree: {matched}/{len(results)}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} unmatched section_numbers: {unmatched}")

    # ── Save hierarchical output JSON ──────────────────────────────────────
    save_json_file(data, args.output)
    print(f"\nHierarchical output saved → {args.output}")


if __name__ == "__main__":
    main()