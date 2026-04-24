import pandas as pd
import json
from src.compare_instructions.llm_call import llm_call  

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
                "category_tags": [],
                "pattern": ""
            }

def analyze_row(row):
    prompt = build_prompt(row)

    response = llm_call(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0  
    )

    parsed = safe_parse(response)

    # retry once if empty
    if not parsed.get("missing_elements"):
        response = llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0
        )
        parsed = safe_parse(response)

    return pd.Series({
        "missing_elements": "; ".join(parsed.get("missing_elements", [])),
        "category_tags": ", ".join(parsed.get("category_tags", [])),
        "pattern": parsed.get("pattern", "")
    })

def process_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    new_cols = df.apply(analyze_row, axis=1)

    df = pd.concat([df, new_cols], axis=1)

    df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    input_csv = "temp_ingestion.csv"  # replace with your input path
    output_csv = "pattern.csv"  # replace with your desired output path
    result_df = process_csv(input_csv, output_csv)
    print(result_df.head())
    print(f"Processed data saved to {output_csv}")
    

