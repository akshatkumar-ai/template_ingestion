"""
transform.py
------------
Reads the system prompt and input instruction from .txt files and runs the
template-instruction transformation via AWS Bedrock (Claude).

Usage
-----
    python transform.py \
        --system_prompt system_prompt.txt \
        --input_instruction input_instruction.txt \
        [--output output.txt] \
        [--model_id <bedrock-model-arn>]

Arguments
---------
--system_prompt       Path to the .txt file containing the full system prompt
                      (role, task, transformation_rules, example, output_format).
--input_instruction   Path to the .txt file containing the raw input instruction
                      to be transformed.
--output              (Optional) Path to write the transformed output.
                      If omitted, the result is printed to stdout only.
--model_id            (Optional) AWS Bedrock model ARN / inference-profile ID.
                      Defaults to the Claude Sonnet 4 inference profile used in
                      the notebook.
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# AWS Bedrock helpers (ported from the notebook)
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

def build_prompt(system_prompt: str, input_instruction: str) -> str:
    """
    Wraps the raw input_instruction inside the <input_instruction> XML tag
    and appends it to the system prompt, matching the notebook structure.
    """
    return (
        system_prompt.rstrip()
        + "\n\n<input_instruction>\n"
        + input_instruction.strip()
        + "\n</input_instruction>\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = (
    "arn:aws:bedrock:us-east-1:533267065792:"
    "inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
)
DEFAULT_SYSTEM_PROMPT = "system_prompt.txt"
DEFAULT_INPUT_INSTRUCTION = "input_instruction.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform a raw section instruction using AWS Bedrock (Claude)."
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=f"Path to the .txt file containing the system prompt. "
             f"Defaults to: {DEFAULT_SYSTEM_PROMPT}",
    )
    parser.add_argument(
        "--input_instruction",
        default=DEFAULT_INPUT_INSTRUCTION,
        help=f"Path to the .txt file containing the input instruction to transform. "
             f"Defaults to: {DEFAULT_INPUT_INSTRUCTION}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(Optional) Path to write the transformed output. "
             "If omitted, result is printed to stdout only.",
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"AWS Bedrock model ARN or inference-profile ID. "
             f"Defaults to: {DEFAULT_MODEL_ID}",
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
    input_instruction = read_file(args.input_instruction, "Input instruction")

    prompt = build_prompt(system_prompt, input_instruction)

    result = generate(prompt, args.model_id)

    print("\n" + "=" * 60)
    print("TRANSFORMED INSTRUCTION")
    print("=" * 60)
    print(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n[Saved to: {args.output}]")


if __name__ == "__main__":
    main()
