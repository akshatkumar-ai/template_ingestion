"""
batch_transform.py
------------------
Reads the system prompt from .txt file and processes rows n1 to n2 in the
'section_instructions' column of a CSV file sequentially as input instructions,
running the template-instruction transformation via AWS Bedrock (Claude).
Appends the transformed output to a new 'transformed_output' column and the
reasoning to a new 'reasoning' column in the CSV.
Also reads TOC.md and includes it as context in each prompt.
Caches the final prompt and JSON response in outputs/section_number/ folder.

Usage
-----
    python batch_transform.py \
        --system_prompt system_prompt.txt \
        --csv "Protocol Ph2-3 Veriscribe Template Mar 2026 (1) - section-config-extract 1.csv" \
        [--start 1] [--end 10] \
        [--authoring_type protocol|csr|sap] \
        [--model_id <bedrock-model-arn>]

Arguments
---------
--system_prompt       Path to the .txt file containing the full system prompt
                      (role, task, transformation_rules, example, output_format).
--csv                 Path to the CSV file containing section instructions.
                      The file will be modified in place with new 'transformed_output' and 'reasoning' columns.
--start               (Optional) Starting row number (1-based) to process. Defaults to 1.
--end                 (Optional) Ending row number (1-based) to process. Defaults to all rows.
--authoring_type      (Optional) Type of authoring: 'protocol', 'csr', or 'sap'. 
                      Determines which source documents are used as anchor and secondary sources.
                      Defaults to 'protocol'.
--model_id            (Optional) AWS Bedrock model ARN / inference-profile ID.
                      Defaults to the Claude Sonnet 4 inference profile.
"""

import argparse
import csv
import json
import os
import sys
import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    """Load configuration from config.yaml in the repository root."""
    # batch_transform.py is at src/instruction_transform/batch_transform.py
    # config.yaml is at the root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "...", ".."))
    config_path = os.path.join(repo_root, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config, repo_root


def resolve_path(relative_path: str, repo_root: str) -> str:
    """Resolve a relative path from the config to an absolute path."""
    return os.path.join(repo_root, relative_path)


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
# CLI
# ---------------------------------------------------------------------------

def load_defaults_from_config():
    """Load default values from config.yaml"""
    try:
        config, repo_root = load_config()
        instruction_transform_cfg = config.get("INSTRUCTION_TRANSFORM", {})
        
        return {
            "system_prompt": resolve_path(instruction_transform_cfg.get("system_prompt", "prompts/system_prompt.txt"), repo_root),
            "csv": resolve_path(instruction_transform_cfg.get("csv", "data/temp_ingestion.csv"), repo_root),
            "toc": resolve_path(instruction_transform_cfg.get("toc", "data/TOC.md"), repo_root),
            "model_id": instruction_transform_cfg.get("model_id", "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"),
            "outputs": resolve_path(instruction_transform_cfg.get("outputs", "outputs"), repo_root),
            "repo_root": repo_root,
        }
    except Exception as e:
        print(f"Warning: Could not load config - {e}", file=sys.stderr)
        # Fallback to relative defaults
        return {
            "system_prompt": "prompts/system_prompt.txt",
            "csv": "data/temp_ingestion.csv",
            "toc": "data/TOC.md",
            "model_id": "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
            "outputs": "outputs",
            "repo_root": None,
        }


def parse_args():
    defaults = load_defaults_from_config()
    
    parser = argparse.ArgumentParser(
        description="Batch transform section instructions from CSV using AWS Bedrock (Claude) and append to CSV."
    )
    parser.add_argument(
        "--system_prompt",
        default=defaults["system_prompt"],
        help=f"Path to the .txt file containing the system prompt. "
             f"Defaults to: {defaults['system_prompt']}",
    )
    parser.add_argument(
        "--csv",
        default=defaults["csv"],
        help=f"Path to the CSV file containing section instructions. "
             f"The file will be modified in place with new 'transformed_output' and 'reasoning' columns. "
             f"Defaults to: {defaults['csv']}",
    )
    parser.add_argument(
        "--toc",
        default=defaults["toc"],
        help=f"Path to the TOC.md file to use as context. "
             f"Defaults to: {defaults['toc']}",
    )
    parser.add_argument(
        "--outputs",
        default=defaults["outputs"],
        help=f"Directory to store prompt and response outputs. "
             f"Defaults to: {defaults['outputs']}",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting row number (1-based) to process. Defaults to 1.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending row number (1-based) to process. Defaults to all rows.",
    )
    parser.add_argument(
        "--model_id",
        default=defaults["model_id"],
        help=f"AWS Bedrock model ARN or inference-profile ID. "
             f"Defaults to: {defaults['model_id']}",
    )
    parser.add_argument(
        "--authoring_type",
        default="protocol",
        choices=["protocol", "csr", "sap"],
        help="Authoring type: 'protocol', 'csr', or 'sap'. Defaults to 'protocol'.",
    )
    return parser.parse_args()


def read_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()

    system_prompt = read_file(args.system_prompt, "System prompt")

    toc_content = read_file(args.toc, "TOC")

    os.makedirs(args.outputs, exist_ok=True)

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Add new columns if not present
    if "transformed_output" not in fieldnames:
        fieldnames = list(fieldnames) + ["transformed_output"]
    if "reasoning" not in fieldnames:
        fieldnames = list(fieldnames) + ["reasoning"]

    # Determine rows to process
    start = max(1, args.start)  # Ensure at least 1
    end = args.end
    if end is not None and end < start:
        print("ERROR: --end must be >= --start", file=sys.stderr)
        sys.exit(1)
    start_idx = start - 1  # 0-based
    end_idx = min(end, len(rows)) if end is not None else len(rows)
    rows_to_process = rows[start_idx:end_idx]

    print(f"Processing rows {start} to {end_idx} (1-based)")

    client = get_bedrock_client()

    # Process each selected row
    for row in rows_to_process:
        section_number = row.get("section_number", "").strip()
        section_title = row.get("section_title", "").strip()
        section_instructions = row.get("template_instructions", "").strip()
        if not section_instructions:
            row["transformed_output"] = ""
            row["reasoning"] = ""
            continue

        print(f"\nProcessing section {section_number}...")

        section_context = (
            f"section_number: {section_number}\n"
            f"section_title: {section_title}"
        )

        prompt = build_prompt(system_prompt, section_context, toc_content, section_instructions, args.authoring_type)
        result = generate(prompt, args.model_id, client)

        # Cache prompt and response
        section_folder = os.path.join(args.outputs, section_number)
        os.makedirs(section_folder, exist_ok=True)
        with open(os.path.join(section_folder, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(os.path.join(section_folder, "response.json"), "w", encoding="utf-8") as f:
            f.write(result)

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
            
            data = json.loads(clean_result)
            row["transformed_output"] = data.get("transformed_instruction", "")
            row["reasoning"] = data.get("reasoning", "")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for section {section_number}: {e}")
            row["transformed_output"] = result  # Fallback to raw result
            row["reasoning"] = ""
        print(f"Transformed section {section_number}")

    # Write back to CSV
    with open(args.csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated CSV: {args.csv}")


if __name__ == "__main__":
    main()