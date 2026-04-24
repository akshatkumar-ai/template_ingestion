import os
import json




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


def generate(prompt: str, model_id: str, client=None, temperature: float = 0.1, max_tokens: int = 8192, top_p: float = 1.0, stop: list = None, **kwargs) -> str:
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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt.strip()}],
                }
            ],
        }
        if stop:
            native_request["stop_sequences"] = stop

        response = client.invoke_model(
            modelId=model_id, body=json.dumps(native_request)
        )
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]

    except Exception as e:
        print(e)
        raise ValueError("Error in generation!")

DEFAULT_MODEL_ID = (
    "arn:aws:bedrock:us-east-1:533267065792:"
    "inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
)

def llm_call(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0,
    max_tokens: int = 2000,
    top_p: float = 1.0,
    stop: list = None,
    **kwargs
) -> str:
    """
    Generic LLM call wrapper.

    Args:
        prompt (str): User/content prompt
        system_prompt (str): System-level instruction
        temperature (float): Controls randomness (0 = deterministic)
        max_tokens (int): Max tokens in output
        top_p (float): nucleus sampling
        stop (list): stop sequences
        **kwargs: future extensions

    Returns:
        str: model response
    """

    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)

    prompt = prompt.strip()
    system_prompt = system_prompt.strip()

    # Combine prompts cleanly
    full_prompt = ""

    if system_prompt:
        full_prompt += f"[SYSTEM]\n{system_prompt}\n\n"

    full_prompt += f"[USER]\n{prompt}"

    response = generate(
        full_prompt,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        **kwargs
    )

    return response.strip()
