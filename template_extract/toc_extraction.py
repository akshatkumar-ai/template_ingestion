import sys
import os
import time
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from pypdf import PdfReader
from tqdm import tqdm
import json
import re
from pydoc import importfile
from typing import List, Dict, Set, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from general_prompts import extract_toc_prompt_mardown_json

# Add the parent directory to the path to access utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from general import (
    get_config,
    generate,
    get_bedrock_client,
    write_json_to_file,
    read_file_contents,
)
from logger_config import logger


# HELPER FUNCTIONS
def get_toc_page(reader):
    """Find the page number containing the table of contents.

    Parameters
    ----------
    reader : PyPDF2.PdfReader
        PDF reader object containing the document to search.

    Returns
    -------
    int
        The page index (0-based) containing the table of contents.
        If no TOC is found, function will return nothing.

    Notes
    -----
    The function searches for 'table of contents' in each page's text,
    ignoring case and extra spaces.

    Examples
    --------
    >>> reader = PyPDF2.PdfReader('document.pdf')
    >>> toc_page = get_toc_page(reader)
    >>> print(f'TOC in page: {toc_page}')
    TOC in page: 2
    """
    for idx, page in enumerate(reader.pages):
        if "table of contents" in page.extract_text().lower().replace("  ", " "):
            print(f"TOC in page: {idx}")
            return idx


def extract_template_text(template_path: str) -> str:
    """
    Extracts text content from a PDF template starting from a specific page.

    Args:
        template_path (str): Path to the template PDF file.

    Returns:
        str: Concatenated text from all pages starting from the TOC (table of contents) page onward.

    Note:
        - This function depends on an external `get_toc_page(template_reader)` function that returns
          the index of the page from which to start extraction.
        - Make sure tqdm, pypdf, and get_toc_page are available in your environment.
    """
    try:
        logger.info(f"Reading template PDF: {template_path}")
        template_reader = PdfReader(template_path)
        start_index = get_toc_page(template_reader)  # Assumed to be defined elsewhere

        logger.info(f"Starting text extraction from page index: {start_index}")

        template_text = ""
        for page in tqdm(
            template_reader.pages[start_index:], desc="Extracting template text"
        ):
            text = page.extract_text()
            if text:
                template_text += text

        logger.info("Template text extraction completed.")
        return template_text

    except Exception as e:
        logger.error(f"Error during template text extraction: {e}", exc_info=True)
        raise e


def parse_toc(input_string: str) -> str:
    """
    Parses a table of contents string formatted with pipes (|) into a JSON-formatted string.

    Args:
        input_string (str): A multi-line string representing a table with headers
                            |section_number|section_name| and subsequent data rows.

    Returns:
        str: A JSON-formatted string representing the parsed table of contents in the format:
             {
               "TOC": [
                   {"section_number": "1", "section_name": "Introduction"},
                   {"section_number": "1.1", "section_name": "Clinical investigation rationale"},
                   ...
               ]
             }
    """
    logger.info("Starting to parse TOC string.")

    try:
        lines = input_string.strip().split("\n")[1:]  # Skip header
        toc = []

        for line in lines:
            parts = line.strip("|").split("|")
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line: {line}")
                continue
            section_number = parts[0].strip()
            section_name = parts[1].strip()
            toc.append({"section_number": section_number, "section_name": section_name})
            logger.debug(f"Added section: {section_number} - {section_name}")

        # result = {"TOC": toc}
        # json_output = json.dumps(result, indent=2)
        json_output = json.dumps(toc, indent=2)
        logger.info("Successfully parsed TOC string.")
        return json_output

    except Exception as e:
        logger.error(f"Error while parsing TOC: {e}")
        raise


def extract_result_content(text, tag):
    """
    Extracts content inside the <result>...</result> tags using regex.

    Args:
        text (str): The input string containing <result> tags.
        tag (str): the input tag (for e.g, result)

    Returns:
        str: The content inside the <result> tag, or None if not found.
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_toc_from_template(
    template_text: str, prompt: str, model_id: str, client
) -> str:
    """
    Uses Language model to extract the table of contents (TOC) from the given template text.

    Args:
        template_text (str): The raw text extracted from a template PDF.
        prompt (str): Prompt to guide the LLM on what to extract.
        model_id (str): Identifier for the model to use.
        client: The client used for generating the response.

    Returns:
        str: The LLM response, which should ideally be a parsed TOC.

    """
    max_retries = 3
    retry_delay = 0.1
    for attempt in range(max_retries):
        try:
            logger.info(f"Started extracting TOC from template using LLM")
            prompt_template = prompt.format(text=template_text)
            response = generate(prompt_template, client=client, model_id=model_id)
            response = response.replace("```json", "").replace("```", "")
            logger.info("TOC extraction successful.")
            return response
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} for extracting TOC from template failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                logger.error(
                    f"All attempts for extracting TOC from template failed.{e}"
                )
                raise e


def extract_all_headings(markdown_text):
    """
    Extract all headings from markdown text based on section number and name.
    Returns a list of tuples: (start_pos, end_pos, section_number, section_name)
    """
    heading_pattern = re.compile(r"^(#+\s+)(\d+(?:\.\d+)*\.?)\s+(.*)$", re.MULTILINE)
    headings = []
    for match in heading_pattern.finditer(markdown_text):
        start_pos = match.start()
        end_pos = match.end()  # <-- Use this to skip the heading line
        section_number = match.group(2).strip().strip(".")
        section_name = match.group(3).strip()
        headings.append((start_pos, end_pos, section_number, section_name))
    return headings


def convert_section_tuples_to_dicts(section_tuples):
    """
    Convert list of section tuples to list of dictionaries with section_number and section_name.

    Parameters:
        section_tuples (list of tuple): Each tuple should be of the form (start, end, section_number, section_name)

    Returns:
        list of dict: With keys 'section_number' and 'section_name'
    """
    return [
        {"section_number": sec_num, "section_name": sec_name}
        for _, _, sec_num, sec_name in section_tuples
    ]


def normalize_section_number(section_number: Optional[str]) -> Optional[str]:
    """
    Normalize the section number by removing a single trailing '.0' or '.' if present.
    Does not remove '0' if it is the entire section number.

    Parameters:
        section_number (str or None): The section number to normalize.

    Returns:
        str or None: The normalized section number, or None if input is None.
    """
    if isinstance(section_number, str):
        if section_number == "0":
            return section_number  # Don't strip if it's just '0'
        # Remove trailing '.0' only if it's at the end
        if section_number.endswith(".0"):
            return section_number[:-2]
        # Remove trailing '.' if any
        if section_number.endswith("."):
            return section_number[:-1]
    return section_number


def get_truncated_section_number(section_number: str, level: int = 2) -> str:
    """
    Truncate a section number to the specified number of levels.

    Parameters:
        section_number (str): The original section number (e.g., '1.2.1.4').
        level (int): The number of levels to keep (e.g., 1 → '1', 2 → '1.2').

    Returns:
        str: The truncated section number.

    Example:
        get_truncated_section_number("1.2.3.4", level=1) → "1"
        get_truncated_section_number("1.2.3.4", level=2) → "1.2"
    """
    parts = section_number.split(".")
    return ".".join(parts[:level]) if len(parts) >= level else section_number


def build_section_number_set_and_max(
    toc_list: List[Dict[str, str]],
) -> Tuple[Set[str], Optional[str]]:
    """
    Build a set of normalized section numbers and find the highest top-level section number.

    Parameters:
        toc_list (list of dict): List of section dictionaries with 'section_number'.

    Returns:
        tuple:
            - set of normalized section numbers (str)
            - max top-level section number (str), or None if not found
    """
    normalized_set = set()
    top_levels = []

    for item in toc_list:
        section_number = item.get("section_number")
        if section_number:
            normalized = normalize_section_number(section_number)
            normalized_set.add(normalized)

            top_level = get_truncated_section_number(section_number, level=1)
            normalized_top = normalize_section_number(top_level)
            try:
                top_levels.append(int(normalized_top))
            except ValueError:
                pass  # Skip non-numeric top-level entries like 'APPENDIX A'

    max_top_level = str(max(top_levels)) if top_levels else None
    return normalized_set, max_top_level


def safe_sort_key(section_number: str):
    """
    Generates a sorting key for a section number string that may include both numeric and non-numeric parts.

    This function is useful when section numbers are hierarchical and may contain non-numeric suffixes
    (e.g., "10.2", "10.2.1", "Appendix A"). It ensures that numeric parts are sorted numerically and
    non-numeric parts lexicographically, while preserving the hierarchy structure.

    Args:
        section_number (str): The section number string (e.g., "2.1", "Appendix A").

    Returns:
        List[Tuple[int, Union[int, str]]]: A list of tuples to be used as a sort key.
    """
    parts = section_number.split(".")
    key = []
    for part in parts:
        try:
            numeric_part = int(part)
            key.append((0, numeric_part))  # numeric part
        except ValueError:
            key.append((1, part.lower()))  # non-numeric part like 'APP'
    logger.debug(f"Generated sort key {key} for section number '{section_number}'")
    return key


def sort_sections_by_number(section_list):
    return sorted(section_list, key=lambda x: safe_sort_key(x["section_number"]))


def remove_unsorted_sections(section_blocks):
    """
    Removes section blocks where section_number is not in strictly increasing order.
    Assumes section_number is in a dotted string format like "1", "1.1", "1.2.1", etc.
    """

    def to_tuple(section_number):
        return tuple(map(int, section_number.strip(".").split(".")))

    cleaned_blocks = []
    prev_tuple = ()
    for block in section_blocks:
        current_tuple = to_tuple(block["section_number"])
        if current_tuple > prev_tuple:
            cleaned_blocks.append(block)
            prev_tuple = current_tuple
        else:
            # Skip unsorted or duplicate section
            continue
    return cleaned_blocks


def merge_toc_lists(
    llm_toc_list: List[Dict[str, str]], regex_toc_list: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Merge entries from `regex_toc_list` into `llm_toc_list` if their normalized
    section numbers are not already present in `llm_toc_list`.

    Both lists should contain dictionaries of the form:
        {
            "section_number": "1.2.1",
            "section_name": "PART A: Dose Escalation Phase"
        }

    Parameters:
        llm_toc_list (list of dict): The primary TOC list to be merged into.
        regex_toc_list (list of dict): The secondary TOC list to merge from.

    Returns:
        list of dict: The updated `llm_toc_list` with new entries added.
    """
    existing_sections, max_existing_section = build_section_number_set_and_max(
        llm_toc_list
    )

    for item in regex_toc_list:
        sec_num = normalize_section_number(item.get("section_number"))
        truncated_sec_num = get_truncated_section_number(sec_num, level=1)

        # check whether max_section obtained from LLM_toc list. If max_section is less than the one we are adding from regex_toc then we are not adding that section
        add_section = True
        try:
            if int(max_existing_section) < int(truncated_sec_num):
                logger.warning("❌ WARNING:")
                logger.warning(
                    f"The max section number present in toc is {int(max_existing_section)} is less than section {int(truncated_sec_num)} present in regex."
                )
                add_section = False
        except:
            pass

        if sec_num and sec_num not in existing_sections and add_section:
            logger.info("✅ SUCCESS:")
            logger.info(f"Added section {item} inside llm toc list from regex toc list")
            llm_toc_list.append(item)
            existing_sections.add(sec_num)  # Prevent future duplicates

    sorted_llm_toc_list = sort_sections_by_number(llm_toc_list)

    return sorted_llm_toc_list


def toc_extraction(
    template_pdf_path: str,
    markdown_file_path: str,
    model_id: str,
    toc_json_llm_extract: str,
    toc_json_llm_regex_extract: str,
) -> None:
    """
    Extracts Table of Contents (TOC) using both LLM and regex methods from a given template and markdown file.

    Args:
        template_pdf_path: Path to the PDF file used as a template.
        markdown_file_path: Path to the markdown file for regex extraction.
        model_id: Model ID used for LLM extraction.
        toc_json_llm_extract (str): Output path for storing the LLM-extracted TOC.
        toc_json_llm_regex_extract (str): Output path for storing the merged TOC (LLM + regex).
    """
    logger.info("Starting template to JSON conversion pipeline...")

    # Read the markdown file
    markdown_template_text = read_file_contents(markdown_file_path)

    # Initialize Bedrock client
    client = get_bedrock_client()
    logger.info("Bedrock client initialized")

    # Extract text from template
    logger.info(f"Extracting template text from: {template_pdf_path}")
    template_text = extract_template_text(template_pdf_path)
    logger.debug(f"Extracted template text (truncated): {template_text[:200]}")

    # Extract TOC from template using the model
    logger.info("Sending request to extract TOC from template...")
    response = extract_toc_from_template(
        template_text, extract_toc_prompt_mardown_json, model_id, client
    )
    logger.debug(f"Raw response from model (truncated): {response[:200]}")

    # Extract result content
    response = extract_result_content(response, "result")
    logger.debug(f"Extracted result content (truncated): {response[:200]}")

    # Parse TOC
    logger.info("Parsing TOC from response.")
    llm_toc_list = eval(parse_toc(response))
    logger.debug(f"Parsed TOC: {llm_toc_list}")

    # Write LLM TOC output
    write_json_to_file(toc_json_llm_extract, llm_toc_list)

    # Extracting headings using regex
    logger.info("Extracting headings from template text using regex")
    headings = extract_all_headings(markdown_template_text)
    logger.info(f"Extracted {len(headings)} headings from template")

    # Process TOC in list format
    logger.info("Converting section tuples to dictionaries")
    regex_toc_list = convert_section_tuples_to_dicts(headings)
    logger.info(f"Raw regex TOC list contains {len(regex_toc_list)} items")

    logger.info("Removing unsorted sections")
    regex_toc_list = remove_unsorted_sections(regex_toc_list)
    logger.info(f"Filtered regex TOC list contains {len(regex_toc_list)} valid items")

    # Merge TOC lists
    logger.info("Merging LLM TOC with regex TOC")
    updated_llm_toc_list = merge_toc_lists(llm_toc_list[:], regex_toc_list)
    logger.info(f"Merged TOC contains {len(updated_llm_toc_list)} total items")

    # Write merged output
    write_json_to_file(toc_json_llm_regex_extract, updated_llm_toc_list)

    logger.info("TOC extraction completed successfully ✅")


if __name__ == "__main__":
    # Load configuration values
    try:
        script_dir = Path(__file__).resolve().parent
        inputs_dir = script_dir / "inputs"

        template_pdf_path = inputs_dir / '53-ph-2-3-protocol_template.pdf'
        markdown_file_path = inputs_dir / '53-ph-2-3-protocol_template.md'
        model_id = 'arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'
        output_directory = script_dir / 'outputs'
    except KeyError as e:
        logger.error(f"❌ Missing configuration key: {e}")
        sys.exit()

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct full output paths
    toc_json_llm_extract = os.path.join(output_directory, "toc_llm_extract.json")
    toc_json_llm_regex_extract = os.path.join(
        output_directory, "toc_llm_regex_extract.json"
    )

    # Run TOC extraction
    toc_extraction(
        template_pdf_path=template_pdf_path,
        markdown_file_path=markdown_file_path,
        model_id=model_id,
        toc_json_llm_extract=toc_json_llm_extract,
        toc_json_llm_regex_extract=toc_json_llm_regex_extract,
    )