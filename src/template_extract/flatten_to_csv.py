import json
import csv
import argparse


def flatten_json_to_csv(json_path, csv_path, toc_path):
    """
    Converts the hierarchical section_instructions.json to a flat CSV format
    maintaining the order from toc_llm_extract.json.
    
    Args:
        json_path (str): Path to the section_instructions.json file
        csv_path (str): Path to the output CSV file
        toc_path (str): Path to the toc_llm_extract.json file for ordering
    """
    # Load TOC data for ordering
    with open(toc_path, 'r') as f:
        toc_data = json.load(f)
    
    # Load section instructions
    with open(json_path, 'r') as f:
        instructions_data = json.load(f)
    
    # Create a mapping from section_number to instructions
    instructions_map = {}
    
    # Build the mapping from the hierarchical structure
    for section in instructions_data.get('sections', []):
        section_number = section.get('section_number', '')
        instructions_map[section_number] = {
            'section_name': section.get('section_name', ''),
            'instructions': section.get('instructions', ''),
            'section_dependency': section.get('section_dependency', '')
        }
        
        # Add subsections
        for subsection in section.get('subsections', []):
            subsection_number = subsection.get('section_number', '')
            instructions_map[subsection_number] = {
                'section_name': subsection.get('section_name', ''),
                'instructions': subsection.get('instructions', ''),
                'section_dependency': subsection.get('section_dependency', '')
            }
    
    # Create rows in TOC order
    rows = []
    for toc_item in toc_data:
        section_no = toc_item.get('section_number', '')
        section_name = toc_item.get('section_name', '')
        
        # Get instructions if available
        instructions = ''
        dep_value = ''
        if section_no in instructions_map:
            instructions = instructions_map[section_no].get('instructions', '')
            dep_value = instructions_map[section_no].get('section_dependency', '')
        
        # Normalize dependencies: handle dict, list, string, or None
        if isinstance(dep_value, dict):
            dep_list = dep_value.get('dependencies', [])
        elif isinstance(dep_value, list):
            dep_list = dep_value
        elif isinstance(dep_value, str) and dep_value:
            dep_list = [dep_value]
        else:
            dep_list = []
        
        row = {
            'section_no': section_no,
            'section_name': section_name,
            'section_instructions': instructions,
            'section_dependency': dep_list
        }
        rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = [
            'section_no',
            'section_name',
            'section_instructions',
            'section_dependency'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"CSV file created successfully at: {csv_path}")
        print(f"Total rows written: {len(rows)}")
    else:
        print("No data to write to CSV")

def main():
    parser = argparse.ArgumentParser(
        description="Flatten section_instructions.json to CSV using TOC ordering."
    )
    parser.add_argument(
        "--json_path", required=True,
        help="Path to the section_instructions.json file"
    )
    parser.add_argument(
        "--csv_path", required=True,
        help="Path to the output CSV file"
    )
    parser.add_argument(
        "--toc_path", required=True,
        help="Path to the toc_llm_extract.json file for ordering"
    )
    args = parser.parse_args()
    flatten_json_to_csv(args.json_path, args.csv_path, args.toc_path)


if __name__ == "__main__":
    main()
