import json
import sys
from pathlib import Path

import yaml

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.template_extract.template import extract_section_instructions
from src.template_extract.flatten_to_csv import flatten_json_to_csv
from src.template_extract.toc_extraction import toc_extraction


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def resolve_path(path_value, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


if __name__ == "__main__":
    config_path = repo_root / "config.yaml"
    config = load_config(config_path)
    config_base = config_path.parent
    
    # Access TEMPLATE_EXTRACT configuration
    template_config = config.get("TEMPLATE_EXTRACT", {})

    template_pdf_path = resolve_path(template_config["template_pdf_path"], config_base)
    markdown_file_path = resolve_path(template_config["markdown_file_path"], config_base)
    model_id = template_config.get(
        "model_id",
        "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    )
    toc_json_llm_extract = resolve_path(template_config["toc_json_llm_extract"], config_base)
    toc_json_llm_regex_extract = resolve_path(template_config["toc_json_llm_regex_extract"], config_base)
    output_path = resolve_path(template_config["section_instructions_json"], config_base)
    csv_output_path = resolve_path(template_config["section_instructions_csv"], config_base)

    toc_extraction(
        template_pdf_path=str(template_pdf_path),
        markdown_file_path=str(markdown_file_path),
        model_id=model_id,
        toc_json_llm_extract=str(toc_json_llm_extract),
        toc_json_llm_regex_extract=str(toc_json_llm_regex_extract),
    )

    toc_path = str(toc_json_llm_regex_extract)
    with open(toc_path, "r", encoding="utf-8") as f:
        toc_data = json.load(f)

    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    extract_section_instructions(toc_data, markdown_text, str(output_path))
    flatten_json_to_csv(str(output_path), str(csv_output_path), toc_path)
