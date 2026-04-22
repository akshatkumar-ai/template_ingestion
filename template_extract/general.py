import os, sys

sys.path.append(os.path.join(os.path.abspath(__file__), "../../"))
import json
from dotenv import load_dotenv
import yaml
import boto3
import json
from logger_config import logger
from typing import List, Dict, Any, Optional, Union
import time
import json

import re
import pandas as pd

VALID_MODEL_IDS = [
    "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
]


def get_config(parent_path: Optional[str] = None) -> dict:
    """
    Load and return configuration from YAML file.

    Parameters
    ----------
    parent_path : str, optional
        Specific section of the config to return. If None, returns entire config.

    Returns
    -------
    dict
        Configuration dictionary or specific section if parent_path is provided.

    Notes
    -----
    Loads configuration from CONFIG_PATH. If parent_path is specified,
    returns only that section of the configuration.
    """
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    if parent_path:
        return config[parent_path]
    else:
        return config


def get_bedrock_client() -> boto3.client:
    """
    Create and return an AWS Bedrock client.

    Returns
    -------
    boto3.client
        Configured AWS Bedrock client instance.

    Raises
    ------
    ValueError
        If required AWS credentials are missing or if client creation fails.

    Notes
    -----
    Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
    to be set in environment variables.
    """
    print("*" * 50)
    default_env_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    )
    env_path = os.getenv("secret_env", default_env_path)
    print("Connecting to AWS BEDROCK....")

    load_dotenv(env_path)

    # Fetch API keys and other sensitive information from environment variables
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

  
    AWS_REGION = 'us-east-1'

    # Check if the required environment variables are set
    if not AWS_ACCESS_KEY_ID:
        raise ValueError("AWS_ACCESS_KEY_ID is missing in the environment variables.")

    if not AWS_SECRET_ACCESS_KEY:
        raise ValueError(
            "AWS_SECRET_ACCESS_KEY is missing in the environment variables."
        )

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


def generate(prompt: str, model_id: str, client: Optional[boto3.client] = None) -> str:
    """
    Generate text using AWS Bedrock model.

    Parameters
    ----------
    prompt : str
        The input prompt to generate text from.
    model_id : str
        The ID of the model to use for generation. Must be in VALID_MODEL_IDS.
    client : boto3.client, optional
        AWS Bedrock client instance. If None, a new client will be created.

    Returns
    -------
    str
        Generated text response from the model.

    Raises
    ------
    ValueError
        If model_id is invalid, client creation fails, or generation fails.

    Notes
    -----
    Uses Claude 3 Sonnet model by default. Temperature is set to 0.1 for more
    deterministic outputs. Max tokens is set to 8192.
    """
    # Load environment variables from the path specified in the environment,
    # or fall back to the default .env file in the repository root.
    default_env_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    )
    env_path = os.getenv("secret_env", default_env_path)
    load_dotenv(env_path)

    if model_id not in VALID_MODEL_IDS:  # Add other valid model IDs here
        raise ValueError(
            f"Unknown model_name '{model_id}', please modify the function 'generate' in utils.py"
        )

    print(f"Generating response using {model_id}")
    if client is None:
        try:
            client = get_bedrock_client()
        except:
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

        # Convert the native request to JSON.
        request = json.dumps(native_request)
        response = client.invoke_model(modelId=model_id, body=request)

        # Decode the response body.
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]

    except Exception as e:
        print(e)
        raise ValueError("Error in generation!")





def write_json_to_file(file_path: str, data: dict, indent: int = 4) -> None:
    """
    Write JSON data to a file with proper error handling.

    Parameters
    ----------
    file_path : str
        Path to the JSON file to be written.
    data : dict
        Data which needs to be written as JSON.
    indent : int, optional
        Number of spaces for indentation. Defaults to 4.

    Raises
    ------
    IOError
        If there's an error writing to the file.
    TypeError
        If the data can't be serialized to JSON.
    Exception
        For any other unexpected errors.

    Examples
    --------
    >>> data = {'key': 'value'}
    >>> write_json_to_file('output.json', data)
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent)
        logger.info(f"Successfully wrote JSON data to {file_path}")
    except IOError as e:
        logger.error(f"Failed to write to file {file_path}: {str(e)}")
        raise
    except TypeError as e:
        logger.error(f"Data serialization failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while writing JSON to file: {str(e)}")
        raise


def read_file_contents(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read and return the contents of a file with proper error handling.

    Parameters
    ----------
    file_path : str
        Path to the file to be read.
    encoding : str, optional
        File encoding to use. Defaults to "utf-8".

    Returns
    -------
    str
        Contents of the file as a string.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    PermissionError
        If lacking permissions to read the file.
    UnicodeDecodeError
        If there's an encoding mismatch.
    Exception
        For any other unexpected errors.

    Examples
    --------
    >>> content = read_file_contents("template.md")
    >>> print(content[:100])  # Print first 100 characters
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path} - {str(e)}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied reading file: {file_path} - {str(e)}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {str(e)}")
        raise






