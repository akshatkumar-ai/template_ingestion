from llm_call import llm_call
import json
import pandas as pd

CLUSTERS = [
    "Pharmacodynamics & Mechanism of Action",
    "Clinical Pharmacology & Dosing Rationale",
    "Safety Profile & Risk Management",
    "Study Population & Patient Justification",
    "Regulatory, Compliance & Oversight",
    "Trial Administration & Operational Standards",
    "Clinical Trial Context & Background Evidence",
    "Trial Logistics & Procedures"
]

VALID_CLUSTERS = set(CLUSTERS)

cluster_system_prompt = """
You are a classification system.

Your task is to assign a given clinical category to exactly ONE cluster from a predefined list.

CLUSTERS:
1. Pharmacodynamics & Mechanism of Action
2. Clinical Pharmacology & Dosing Rationale
3. Safety Profile & Risk Management
4. Study Population & Patient Justification
5. Regulatory, Compliance & Oversight
6. Trial Administration & Operational Standards
7. Clinical Trial Context & Background Evidence
8. Trial Logistics & Procedures

RULES:
- You MUST select ONLY one cluster from the list above
- Do NOT create new cluster names
- Do NOT modify cluster names
- Choose the closest semantic match
- If unsure, choose the best approximate fit

OUTPUT FORMAT (strict JSON):
{
  "cluster": "<one exact cluster name>"
}
"""

def build_cluster_prompt(category):
    return f"""
Clinical Category:
{category}

Assign this category to the most appropriate cluster.
"""

def assign_cluster_llm(category):
    prompt = build_cluster_prompt(category)

    response = llm_call(
        prompt=prompt,
        system_prompt=cluster_system_prompt,
        temperature=0
    )

    cluster = safe_parse_cluster(response)

    # enforce strict match
    if cluster not in VALID_CLUSTERS:
        return "UNMAPPED"

    return cluster


def safe_parse_cluster(response):
    try:
        parsed = json.loads(response)
        return parsed.get("cluster", "")
    except:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            parsed = json.loads(response[start:end])
            return parsed.get("cluster", "")
        except:
            return ""

def clean_cluster(output):
    return output if output in VALID_CLUSTERS else "UNMAPPED"



def process_clustering(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # 🔥 Step 1: unique categories only
    unique_categories = df["clinical_category"].dropna().unique()

    print(f"Unique categories: {len(unique_categories)}")

    # 🔥 Step 2: build mapping
    mapping = {}

    for i, cat in enumerate(unique_categories):
        print(f"Processing {i+1}/{len(unique_categories)}: {cat}")

        cluster = assign_cluster_llm(cat)
        mapping[cat] = cluster

    # 🔥 Step 3: map back
    df["cluster"] = df["clinical_category"].map(mapping)

    # 🔥 Step 4: save
    df.to_csv(output_csv, index=False)

    return df, mapping


if __name__ == "__main__":
    input_csv = "pattern.csv"
    output_csv = "clustered_output.csv"

    df, mapping = process_clustering(input_csv, output_csv)

    print(df.head())
    print(f"Saved to {output_csv}")