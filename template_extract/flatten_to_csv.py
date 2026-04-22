import json
import csv


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
        dependencies = ''
        if section_no in instructions_map:
            instructions = instructions_map[section_no].get('instructions', '')
            dependencies = instructions_map[section_no].get('section_dependency', '')
        
        row = {
            'section_no': section_no,
            'section_name': section_name,
            'section_instructions': instructions,
            'section_dependency': dependencies
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
