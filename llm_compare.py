import pandas as pd
import json
from llm_call import llm_call  

system_prompt = """

    You are comparing human-written instructions (SME) with AI-generated instructions.

Your task is to identify what meaningful instruction elements exist in the SME version but are missing in the AI-generated version.

INPUT:

* Section: {section_number} - {section_title}
* Section Dependencies: {section_dependencies}
* Template Instructions: {template_instructions}
* SME Instructions: {ground_truth}
* AI Instructions: {transformed_output}

TASK:

1. Identify ONLY meaningful missing elements (ignore wording/style differences)
2. Extract exact phrases from SME instructions
3. Classify them into categories:

   * DOMAIN_CONSTRAINT
   * EDGE_CASE_HANDLING
   * CROSS_SECTION_AWARENESS
   * REGULATORY_OR_COMPLIANCE
   * EXECUTION_DETAIL
   * DATA_DEPENDENCY

OUTPUT FORMAT (strict JSON):

{
"missing_elements": ["<text1>", "<text2>"],
"category_tags": ["<category1>", "<category2>"],
"pattern": "<one-line insight>"
}

CONSTRAINTS:

* Do NOT invent anything
* Do NOT add new logic
* Use only SME text
* Keep output concise
"""

new_system_prompt ="""
You are analyzing differences between human-written clinical instructions (SME) and AI-generated instructions.

Your goal is to identify what types of **clinical knowledge or domain expertise** are present in SME instructions but missing in AI-generated instructions.

INPUT:

* Section: {section_number} - {section_title}
* Section Dependencies: {section_dependencies}
* Template Instructions: {template_instructions}
* SME Instructions: {ground_truth}
* AI Instructions: {transformed_output}


TASK:
Step 1 — Identify missing elements
Compare the SME instructions against the AI instructions. Find all meaningful
clinical or domain-specific elements that are present in the SME instructions
but absent from the AI instructions. Ignore stylistic or formatting differences;
focus only on substantive clinical content.
Step 2 — Extract exact phrases
For each missing element, copy the exact phrase or sentence from the SME
instructions. Do not paraphrase or summarize.
Step 3 — Decompose each element into action + object
For each extracted phrase, identify:
- The action being requested (e.g., "describe", "indicate", "explain")
- The object or clinical concept it applies to (e.g., "mechanism of action",
"drug class", "interaction between mechanisms")
Example:
SME instruction:
"Describe comprehensively and with great detail how the investigational drug
or treatment exerts its intended biological effects. Indicate the drug class
or type, and describe the mechanism of action for its drug class or type.
Also describe how two or more mechanisms of action might interact."
Decomposed elements:
- Explanation of how the investigational drug exerts biological effects
- Statement of the drug class or type
- Description of the interaction between different mechanisms of action
Step 4 — Assign a clinical category to each element
Group each element under a category that reflects the underlying clinical or
domain knowledge it represents. Follow these rules:
- Generate categories dynamically based on what you find — do not use a
predefined list
- Categories must reflect clinical or domain meaning (e.g., "Pharmacodynamic
Mechanism Detail"), not document structure (e.g., "Section 3 Content")
- Keep category labels concise but specific enough to be meaningful
Step 5 — Assign a shared cluster label (pattern)
After completing Steps 1–4 for ALL sections, review the clinical_category
values across every section. Identify recurring themes and group them into
a small set of shared cluster labels (aim for 7-10 total clusters across
all sections, but do not force it). There can be many themes for each section.

CRITICAL CONSTRAINTS FOR SPACE (OUTPUT TOKEN LIMIT):
- Be EXTREMELY CONCISE in your descriptions to ensure all records fit in the response.
- Use the EXACT "section_id" provided in the input for mapping.
- Output ONLY a valid JSON list.


OUTPUT FORMAT (strict JSON):


{
"missing_elements": [
{
"text": "<exact SME snippet>",
"clinical_category": "<dynamically generated category>",
"why_missing": "<why AI likely missed this knowledge>"
},
{
"text": "<exact SME snippet>",
"clinical_category": "<dynamically generated category>",
"why_missing": "<why AI likely missed this knowledge>"}
],
"pattern": "<1-2 line summary of clinical knowledge gap>"
}

CONSTRAINTS:

* Do NOT invent elements not present in SME text
* Do NOT use predefined category labels
* Categories must reflect underlying clinical meaning
* Keep categories concise but meaningful
* Keep output strictly in JSON format

"""

def build_prompt(row):
    return f"""
SECTION CONTEXT:
Section: {row['section_number']} - {row['section_title']}
Dependencies: {row['section_dependencies']}

TEMPLATE:
{row['template_instructions']}

SME INSTRUCTIONS:
{row['ground_truth']}

AI GENERATED INSTRUCTIONS:
{row['transformed_output']}
"""

def safe_parse(response):
    try:
        return json.loads(response)
    except:
        try:
            # try to extract JSON substring
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except:
            return {
                "missing_elements": [],
                # "category_tags": [],
                
                "pattern": ""
            }

def analyze_row(row):
    prompt = build_prompt(row)

    response = llm_call(
        prompt=prompt,
        system_prompt=new_system_prompt,
        temperature=0  
    )

    parsed = safe_parse(response)

    # retry once if empty
    if not parsed.get("missing_elements"):
        response = llm_call(
            prompt=prompt,
            system_prompt=new_system_prompt,
            temperature=0
        )
        parsed = safe_parse(response)

    missing_elements = parsed.get("missing_elements", [])

    # 🔥 Build list of rows
    # rows = []
    # for elem in missing_elements:
    #     text = elem.get("text", "").strip()
    #     category = elem.get("clinical_category", "").strip()

    #     if text and category:
    #         rows.append({
    #             "section_number": row["section_number"],
    #             "section_title": row["section_title"],
    #             "missing_element": text,
    #             "clinical_category": category
    #         })
    rows = []
    for elem in missing_elements:
        text = elem.get("text", "").strip()
        category = elem.get("clinical_category", "").strip()

        if text and category:
            rows.append({
                "section_number": row["section_number"],
                "section_title": row["section_title"],
                "missing_element": text,
                "clinical_category": category
            })

    return pd.DataFrame(rows)

def process_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    all_rows = []

    for _, row in df.iterrows():
        result_df = analyze_row(row)
        all_rows.append(result_df)

    final_df = pd.concat(all_rows, ignore_index=True)

    final_df.to_csv(output_path, index=False)

    return final_df




if __name__ == "__main__":
    input_csv = "temp_ingestion.csv"  # replace with your input path
    output_csv = "pattern.csv"  # replace with your desired output path
    result_df = process_csv(input_csv, output_csv)
    print(result_df.head())
    print(f"Processed data saved to {output_csv}")
    
    

