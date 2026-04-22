import json
from pathlib import Path
from template import extract_section_instructions
from flatten_to_csv import flatten_json_to_csv
from toc_extraction import toc_extraction

if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).resolve().parent
    inputs_dir = script_dir / "inputs"
    outputs_dir = script_dir / "outputs"
    
    template_pdf_path = inputs_dir / '53-ph-2-3-protocol_template.pdf'
    markdown_file_path = inputs_dir / '53-ph-2-3-protocol_template.md'
    model_id = 'arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'
    
    toc_json_llm_extract = outputs_dir / "toc_llm_extract.json"
    toc_json_llm_regex_extract = outputs_dir / "toc_llm_regex_extract.json"
    
    # Run TOC extraction first
    toc_extraction(
        template_pdf_path=str(template_pdf_path),
        markdown_file_path=str(markdown_file_path),
        model_id=model_id,
        toc_json_llm_extract=str(toc_json_llm_extract),
        toc_json_llm_regex_extract=str(toc_json_llm_regex_extract),
    )
    
    # Load the extracted TOC data
    toc_path = str(toc_json_llm_regex_extract)
    with open(toc_path, 'r') as f:
        toc_data = json.load(f)
    
    markdown_path = str(markdown_file_path)
    with open(markdown_path, 'r') as f:
        markdown_text = f.read()
    
    output_path = str(outputs_dir / 'section_instructions.json')
    extract_section_instructions(toc_data, markdown_text, output_path)
    
    # Convert JSON to CSV maintaining TOC order
    csv_output_path = str(outputs_dir / 'section_instructions.csv')
    flatten_json_to_csv(output_path, csv_output_path, toc_path)