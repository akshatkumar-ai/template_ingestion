import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from src.template_extract.general import generate


def extract_toc(
    markdown_file_path: str, output_path: str
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extract the complete table of contents from a markdown file.
    Enhanced to ensure ALL sections and subsections are captured.

    Parameters
    ----------
    markdown_file_path : str
        Path to the input markdown file
    output_path : str
        Path where the extracted TOC JSON will be saved

    Returns
    -------
    Tuple[Optional[Dict], Optional[str]]
        A tuple containing:
        - The extracted TOC as a dictionary (None if extraction failed)
        - The raw markdown text (None if file reading failed)

    Notes
    -----
    The function includes multiple fallback methods for parsing the output:
    1. Direct JSON parsing
    2. Regex-based JSON extraction
    3. Code block extraction
    4. Simplified prompt retry
    """
    print("🔍 Extracting Table of Contents...")

    # Read markdown file with proper encoding handling
    try:
        with open(markdown_file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        with open(markdown_file_path, "r", encoding="latin-1") as f:
            markdown_text = f.read()

    # Create enhanced TOC extraction prompt with explicit instructions for completeness
    extract_toc_prompt = """
You are provided a markdown document. Your task is to extract the COMPLETE table of contents in the form of a JSON object.

IMPORTANT: You MUST extract ALL sections and subsections from the document, not just the top-level ones.

Instructions:
1. Look at ALL headings and subheadings which typically use # syntax in markdown (like # Section, ## Subsection)
2. For each heading, determine its hierarchical level (# is level 1, ## is level 2, etc.)
3. Include ALL sections and subsections - be exhaustive and complete
4. Include ALL Appendices - these are extremely important. For appendices:
a. Use the appendix identifier (like "Appendix A" or "Appendix 1") as the section_number
b. Include the full appendix name as the section_name
c. Don't miss any appendices even if they have unusual formatting
5. Ignore any non-heading content (tables, lists, etc.)
6. The section numbers should follow hierarchical order (1, 1.1, 1.2, 2, 2.1, etc.)
7. If no explicit section numbers are provided in the markdown, infer them based on heading hierarchy
8. You must return valid JSON with the following structure:

{"TOC":[{"section_number":"1","section_name":"Introduction"},{"section_number":"1.1", "section_name":"Study Rationale"},...,{"section_number":"Appendix A","section_name":"PERFORMANCE STATUS CRITERIA"}]}

Pay special attention to the appendices section if present.

DO NOT include any other text, explanations, or markdown code blocks around the JSON. Return ONLY the JSON.
"""

    # Combine the prompt with the document text
    full_prompt = f"{extract_toc_prompt}\n\nMarkdown Document:\n{markdown_text}"

    # Generate response with increased max tokens to ensure complete TOC is captured
    toc_output = generate(
        full_prompt,
        "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    )

    if toc_output:
        try:
            # First try to parse directly as JSON
            toc_json = json.loads(toc_output)
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(toc_json, outfile, indent=2)
            print(
                f"✅ TOC successfully saved to {output_path} with {len(toc_json.get('TOC', []))} sections"
            )
            return toc_json, markdown_text
        except json.JSONDecodeError:
            # If direct parsing fails, try multiple extraction methods
            print(
                "⚠️ Direct JSON parsing failed, attempting extraction with multiple methods..."
            )

            # Method 1: Extract JSON with improved regex pattern that can handle multi-line JSON
            try:
                json_pattern = re.compile(r'\{[\s\S]*"TOC"\s*:\s*\[[\s\S]*\][\s\S]*\}')
                match = json_pattern.search(toc_output)
                if match:
                    toc_json = json.loads(match.group(0))
                    with open(output_path, "w", encoding="utf-8") as outfile:
                        json.dump(toc_json, outfile, indent=2)
                    print(
                        f"✅ TOC successfully extracted with regex and saved to {output_path} with {len(toc_json.get('TOC', []))} sections"
                    )
                    return toc_json, markdown_text
            except Exception as e:
                print(f"⚠️ Regex JSON extraction failed: {e}")

            # Method 2: Try to extract JSON from code blocks if present
            try:
                if "```json" in toc_output:
                    json_str = toc_output.split("```json")[1].split("```")[0].strip()
                    toc_json = json.loads(json_str)
                    with open(output_path, "w", encoding="utf-8") as outfile:
                        json.dump(toc_json, outfile, indent=2)
                    print(
                        f"✅ TOC successfully extracted from code block and saved to {output_path} with {len(toc_json.get('TOC', []))} sections"
                    )
                    return toc_json, markdown_text
            except Exception as e:
                print(f"⚠️ Code block extraction failed: {e}")

            # Method 3: Fall back to an alternative approach - retry with a simpler prompt
            print("⚠️ All extraction methods failed. Retrying with a simpler prompt...")

            # Create a simpler, more direct prompt
            simple_prompt = """
Extract ALL headings from this markdown document into a JSON table of contents. Include ALL sections, subsections, and appendices.

Format as a simple JSON array of objects with section_number and section_name:
{"TOC":[{"section_number":"1","section_name":"Introduction"},{"section_number":"1.1","section_name":"Study Rationale"},...]}

Only provide the raw JSON, no explanation.
"""
            retry_prompt = f"{simple_prompt}\n\nMarkdown Document:\n{markdown_text}"
            retry_output = generate(
                retry_prompt,
                "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            )

            try:
                # Try to parse the retry output
                toc_json = json.loads(retry_output)
                with open(output_path, "w", encoding="utf-8") as outfile:
                    json.dump(toc_json, outfile, indent=2)
                print(
                    f"✅ TOC successfully extracted with retry and saved to {output_path} with {len(toc_json.get('TOC', []))} sections"
                )
                return toc_json, markdown_text
            except Exception as e:
                print(f"❌ Final TOC extraction attempt failed: {e}")
                print("Malformed response from retry:")
                print(
                    retry_output[:500] + "..."
                    if len(retry_output) > 500
                    else retry_output
                )

            # If all methods fail, save the raw output for debugging and return None
            with open("failed_toc_extraction.txt", "w", encoding="utf-8") as f:
                f.write(toc_output)
            print(
                "❌ Failed to extract TOC. Raw output saved to failed_toc_extraction.txt"
            )
            return None, markdown_text

    # If no output was generated, return None
    print("❌ No output generated for TOC extraction.")
    return None, markdown_text


def extract_section_block(
    markdown_text: str, section_number: str, section_name: str
) -> Optional[str]:
    """
    Extract the content block for a specific section from markdown text.

    This function locates a section by its number and name, then extracts all
    content until the next section of equal or higher level. It handles both
    regular sections and appendices.

    Parameters
    ----------
    markdown_text : str
        The complete markdown text to search through
    section_number : str
        The section number (e.g., "1.1" or "Appendix A")
    section_name : str
        The name/title of the section

    Returns
    -------
    Optional[str]
        The extracted section content, or None if the section was not found

    Notes
    -----
    - Handles both regular sections and appendices
    - Removes content under deeper-level subheadings
    - Uses flexible regex patterns to match various heading formats
    """
    section_name = section_name.strip()
    match = None

    if section_number.startswith("Appendix"):
        # For appendices, look for "X. **Name**" format or "## **Name**" format
        # Try numbered list format first: "X. **Name**"
        appendix_pattern = re.compile(
            rf"^\s*\d+\.\s*\*\*.*?{re.escape(section_name)}.*?\*\*",
            re.MULTILINE | re.IGNORECASE,
        )
        match = appendix_pattern.search(markdown_text)
        
        # If not found, try markdown heading format: "## **Name**"
        if not match:
            heading_pattern = re.compile(
                rf"^\s*#{1,5}\s*\*\*.*?{re.escape(section_name)}.*?\*\*",
                re.MULTILINE | re.IGNORECASE,
            )
            match = heading_pattern.search(markdown_text)
    else:
        # For regular sections, look for markdown headings with # symbols
        # First, try to build a pattern with key words from the section name
        # This makes matching more flexible with special characters
        
        # Try exact match first
        exact_pattern = re.compile(
            rf"^\s*(?:\d+(?:\.\d+)*\.\s*)?(?P<hashes>#+)\s+.*?{re.escape(section_name)}.*?$",
            re.MULTILINE | re.IGNORECASE,
        )
        match = exact_pattern.search(markdown_text)
        
        # If exact match fails, try matching key words from the section name
        if not match:
            # Extract key words (split by spaces, and filter out single chars or special chars only)
            words = [w for w in section_name.split() if len(w) > 1 and re.search(r'[a-zA-Z0-9]', w)]
            if len(words) >= 2:
                # Build pattern that looks for these words in sequence, with flexible spacing
                word_pattern = r'.*?'.join(re.escape(w) for w in words[:3])  # Use first 3 words
                flexible_pattern = re.compile(
                    rf"^\s*(?:\d+(?:\.\d+)*\.\s*)?(?P<hashes>#+)\s+.*?{word_pattern}.*?$",
                    re.MULTILINE | re.IGNORECASE,
                )
                match = flexible_pattern.search(markdown_text)
        
        # Try with Appendix-style name as last resort
        if not match:
            section_number_upper = section_number.upper()
            appendix_name = section_number_upper + "\t" + section_name
            last_pattern = re.compile(
                rf"^\s*(?:\d+(?:\.\d+)*\.\s*)?(?P<hashes>#+)\s+.*?{re.escape(appendix_name)}.*?$",
                re.MULTILINE | re.IGNORECASE,
            )
            match = last_pattern.search(markdown_text)

    if not match:
        print(f"❌ Heading not found for: {section_name}")
        return None

    if section_number.startswith("Appendix"):
        # For appendices, start after the line
        start_idx = match.end()
        # Find next appendix (either numbered or heading format) or end
        next_appendix_pattern = re.compile(
            r"^\s*(?:\d+\.\s*\*\*|#{1,5}\s*\*\*)", re.MULTILINE
        )
        next_match = next_appendix_pattern.search(markdown_text, pos=start_idx)
        end_idx = next_match.start() if next_match else len(markdown_text)
    else:
        start_idx = match.end()
        level = len(match.group("hashes"))

        # Find next heading of same or higher level
        next_heading_pattern = re.compile(
            rf"^\s*(?:\d+(?:\.\d+)*\.\s*)?#{{1,{level}}}\s+", re.MULTILINE
        )
        next_match = next_heading_pattern.search(markdown_text, pos=start_idx)
        end_idx = next_match.start() if next_match else len(markdown_text)

    section_body = markdown_text[start_idx:end_idx]

    if not section_number.startswith("Appendix"):
        # Remove content under subheadings of deeper levels
        sub_heading_pattern = re.compile(
            rf"^\s*(?:\d+(?:\.\d+)*\.\s*)?#{{{level+1},}}.*?$", re.MULTILINE
        )
        sub_matches = list(sub_heading_pattern.finditer(section_body))
        if sub_matches:
            first_sub_start = sub_matches[0].start()
            section_body = section_body[:first_sub_start]

    return section_body.strip()


def _build_nested_sections(sections: List[Dict]) -> List[Dict]:
    """
    Convert a flat list of sections into a nested structure based on section numbers.

    Parameters
    ----------
    sections : List[Dict]
        List of section dictionaries with section_number and section_name

    Returns
    -------
    List[Dict]
        Nested list of sections where subsections are children of their parent sections
    """
    # Sort sections by their numbers to ensure proper nesting
    sorted_sections = sorted(sections, key=lambda x: x["section_number"])
    nested_sections = []
    section_stack = []

    for section in sorted_sections:
        current_level = len(section["section_number"].split("."))
        current_section = section.copy()
        current_section["subsections"] = []

        # Pop sections from stack until we find the parent level
        while (
            section_stack
            and len(section_stack[-1]["section_number"].split(".")) >= current_level
        ):
            section_stack.pop()

        if not section_stack:
            # This is a top-level section
            nested_sections.append(current_section)
        else:
            # This is a subsection
            section_stack[-1]["subsections"].append(current_section)

        section_stack.append(current_section)

    return nested_sections


def extract_section_instructions(
    toc_json: Dict, markdown_text: str, output_path: str
) -> Dict:
    """
    Extract instructions for each section from the markdown text.

    This function processes each section in the TOC and extracts its corresponding
    content from the markdown text. The results are combined into a new JSON
    structure that includes both the TOC information and the section content in a
    nested format that preserves the section hierarchy.

    Parameters
    ----------
    toc_json : Dict
        The table of contents JSON containing section information
    markdown_text : str
        The complete markdown text to extract content from
    output_path : str
        Path where the combined TOC and instructions JSON will be saved

    Returns
    -------
    Dict
        A dictionary containing the sections with their instructions in a nested structure:
        {
            "sections": [
                {
                    "section_number": str,
                    "section_name": str,
                    "instructions": str,
                    "subsections": [
                        {
                            "section_number": str,
                            "section_name": str,
                            "instructions": str,
                            "subsections": []
                        },
                        ...
                    ]
                },
                ...
            ]
        }

    Notes
    -----
    - Creates output directory if it doesn't exist
    - Preserves original section information while adding instructions
    - Handles cases where no instructions are found
    - Maintains section hierarchy in the output structure
    """
    print("🔍 Extracting section instructions...")
    toc_json = {"TOC": toc_json}
    toc_data = toc_json.get("TOC", [])
    sections = []

    for section in toc_data:
        section_number = section.get("section_number")
        section_name = section.get("section_name")

        instructions = extract_section_block(
            markdown_text, section_number, section_name
        )

        # Add instructions to the section object directly
        section_with_instructions = section.copy()
        section_with_instructions["instructions"] = (
            instructions or "No instructions found"
        )
        sections.append(section_with_instructions)

    # Convert flat sections into nested structure
    nested_sections = _build_nested_sections(sections)

    # Return a new TOC json with instructions added in nested format
    toc_with_instructions = {"sections": nested_sections}

    # Save the data with instructions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(toc_with_instructions, outfile, indent=2)

    print(f"✅ Sections with instructions saved to {output_path}")
    return toc_with_instructions

def extract_section_instructions_changed(
    instruction_json: Dict, output_path: str
) -> Dict:
    """
    Extract instructions for each section from the markdown text.

    This function processes each section in the TOC and extracts its corresponding
    content from the markdown text. The results are combined into a new JSON
    structure that includes both the TOC information and the section content in a
    nested format that preserves the section hierarchy.

    Parameters
    ----------
    toc_json : Dict
        The table of contents JSON containing section information
    markdown_text : str
        The complete markdown text to extract content from
    output_path : str
        Path where the combined TOC and instructions JSON will be saved

    Returns
    -------
    Dict
        A dictionary containing the sections with their instructions in a nested structure:
        {
            "sections": [
                {
                    "section_number": str,
                    "section_name": str,
                    "instructions": str,
                    "subsections": [
                        {
                            "section_number": str,
                            "section_name": str,
                            "instructions": str,
                            "subsections": []
                        },
                        ...
                    ]
                },
                ...
            ]
        }

    Notes
    -----
    - Creates output directory if it doesn't exist
    - Preserves original section information while adding instructions
    - Handles cases where no instructions are found
    - Maintains section hierarchy in the output structure
    """
    print("🔍 Extracting section instructions...")

    toc_data = eval(instruction_json)
    sections = []

    for section in toc_data:
        instructions = section.get("instructions")

        # Add instructions to the section object directly
        section_with_instructions = section.copy()
        section_with_instructions["instructions"] = (
            instructions or "No instructions found"
        )
        sections.append(section_with_instructions)

    # Convert flat sections into nested structure
    nested_sections = _build_nested_sections(sections)

    # Return a new TOC json with instructions added in nested format
    toc_with_instructions = {"sections": nested_sections}

    # Save the data with instructions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(toc_with_instructions, outfile, indent=2)

    print(f"✅ Sections with instructions saved to {output_path}")
    return toc_with_instructions



def classify_sections(sections_data, input_document):
    """
    Classify each section and subsection as driver-dependent or driverless.

    Parameters
    ----------
    sections_data : dict
        Section template JSON containing sections and subsections.
    input_document : str
        Text of the input document (Protocol Synopsis or LOI).
        This is the ONLY allowed reference for Driverless sections.
    """

    print("🔍 Classifying sections and subsections as driver-dependent or driverless...")

    def create_classification_prompt(section, input_document):
        """
        Create a prompt for classifying clinical trial protocol sections
        using an explicit Synopsis / LOI input document.
        """

        section_json = json.dumps(section, indent=2)
       
        prompt = (
            """You are an expert in clinical trial protocol authoring and analysis.

Your task is to classify a protocol section as either **Driver-dependent** or **Driverless**.

You will be provided with:
1. A section from a clinical trial protocol template
2. An **input document**, which will be either:
   - a **Protocol Synopsis**, or
   - a **Letter of Intent (LOI)**

The input document is the **ONLY reference document** that may be used to determine whether a section is Driverless.

---

### INPUT DOCUMENT (Synopsis / LOI)
<input_document>
{input_document}
</input_document>

---

### SECTION TO CLASSIFY
""".format(input_document=input_document)
            + section_json
            + """

---

### DEFINITIONS


#### **Driverless Section**
Classify the section as **Driverless** if and only if **MOST OR ALL** of the section’s instructions can be fully authored using **ONLY**:
- The provided **input document (Synopsis / LOI)**, and
- The section’s own instructions, templates, or example text

A section is **Driverless** if ANY of the following conditions apply **AND none of the Driver-dependent conditions apply**:
- The section contains **complete example text** that can be reused verbatim or with minimal edits
- The section provides **templates or boilerplate text** that only require values explicitly present in the input document
- The section instructions are **largely satisfied** by information available in the Synopsis/LOI
- The section contains regulatory, operational, or administrative boilerplate not requiring external references
- The section can be authored **without consulting any document other than the provided input document**

---

#### **Driver-dependent Section**
Classify the section as **Driver-dependent** if **ANY** of the following are true:
- The section requires **information that is expected to be provided by SMEs in advance** (e.g., in Synopsis or LOI) **but is NOT actually present** in the provided input document
  - Example: Exploratory objectives or exploratory endpoints mentioned in instructions but missing from the Synopsis
- The section requires **clinical, scientific, or regulatory judgment** beyond high-level Synopsis/LOI content
- The section requires consultation of **other documents** (e.g., Investigator’s Brochure, literature, prior studies, package inserts)
- The section provides **only open-ended instructions** without reusable example text or templates
- The section instructions are **only partially answered** by the input document (i.e., some but not most instructions are covered)

---

### MANDATORY OVERRIDES (STRICT RULES)

1. **Always Driverless Sections**
   The following sections must **ALWAYS** be classified as **Driverless**, regardless of input content:
   - Primary Objectives
   - Secondary Objectives
   - Exploratory Objectives
   - Primary Endpoints
   - Secondary Endpoints
   - Exploratory Endpoints

2. **Example Text Rule**
   - If the section’s instructions contain placeholders or example markers such as:
     `[Examples]`, `{Example text}`, or similar,
     → classify the section as **Driverless**

3. **Missing Expected Information Rule**
   - If the instructions assume certain information should exist in the Synopsis/LOI **but that information is missing**,  
     → classify the section as **Driver-dependent**

4. **Majority Coverage Rule**
   - A section is **Driverless ONLY if most of its instructional requirements are met** using the input document  
   - If only a minority of instructions can be satisfied, the section is **Driver-dependent**

---

### IMPORTANT CLARIFICATIONS
- Words like *“describe”* or *“specify”* do NOT automatically imply Driver-dependent
- If complete example text or templates are present and can be populated using only the input document, classify as **Driverless**
- If authoring the section requires ANY source beyond the input document, classify as **Driver-dependent**

---
Examples (For GUIDANCE ONLY)

{{
  "sections": [
    {{
      "section_number": "1.1",
      "section_name": "Clinical Investigational Rationale",
      "instructions": "State the problem or question (eg, describe the population, disease, current standard of care, if one exists, and limitations of knowledge or available therapy) and the reason for conducting the clinical trial. What is the overall rationale for conducting the clinical investigation?",
      "classification": "Driver-dependent"
    }},
    {{
      "section_number": "5.4",
      "section_name": "Screen Failures",
      "instructions": "Subjects who are consented to participate in the clinical trial, who do not meet one or more criteria required for participation in the trial during the screening procedures, are considered screen failures. Indicate how screen failures will be handled in the trial, including conditions and criteria upon which re-screening is acceptable, when applicable.\\n\\nExample text provided as a guide, customize as needed:\\n\\nScreen failures are defined as subjects who consent to participate in the clinical trial but are not subsequently randomly assigned to the clinical investigation intervention or entered in the clinical investigation. A minimal set of screen failure information is required to ensure transparent reporting of screen failure subjects, to meet the Consolidated Standards of Reporting Trials (CONSORT) publishing requirements and to respond to queries from regulatory authorities. Minimal information includes demography, screen failure details, eligibility criteria, and any serious adverse event (SAE).\\n\\nIndividuals who do not meet the criteria for participation in this trial (screen failure) because of  may be rescreened. Rescreened subjects should be assigned the same subject number as for the initial screening.",
      "classification": "Driverless"
    }}
  ]
}}

### OUTPUT INSTRUCTION
Respond with **ONLY ONE** of the following values:
- **Driver-dependent**
- **Driverless**

Do not include explanations or additional text.
"""
        )

        return prompt

    def process_sections_recursively(sections):
        """Recursively process sections and subsections."""
        classified_sections = []

        for section in sections:
            classified_section = section.copy()

            classification_prompt = create_classification_prompt(
                section, input_document
            )

            classification = generate(
                classification_prompt,
                "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            )

            # Normalize model output
            if "driverless" in classification.lower():
                driver_classification = "driverless"
            else:
                driver_classification = "driver"

            classified_section["driver_driverless_classification"] = (
                driver_classification
            )

            # Recurse into subsections
            if section.get("subsections"):
                classified_section["subsections"] = process_sections_recursively(
                    section["subsections"]
                )
            else:
                classified_section["subsections"] = []

            classified_sections.append(classified_section)

        return classified_sections

    # Process all sections recursively
    classified_sections = {
        "sections": process_sections_recursively(sections_data.get("sections", []))
    }

    return classified_sections
