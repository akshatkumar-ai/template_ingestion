"""Script to read the JSON template, extract section dependencies, and enrich an existing CSV file with this information."""


import json
import csv

# ── 1. Load JSON ─────────────────────────────────────────────────────────────

with open("protocol_template_with_dependencies.json", "r") as f:
    data = json.load(f)

# ── 2. Flatten JSON into lookup dict ─────────────────────────────────────────

def build_lookup(sections, lookup=None):
    if lookup is None:
        lookup = {}

    for section in sections:
        sec_num = section.get("section_number", "")

        dep_obj = section.get("section_dependency", {})

        dependencies = []
        reasoning = ""

        if isinstance(dep_obj, dict):
            dependencies = dep_obj.get("dependencies", [])
            reasoning = dep_obj.get("reasoning", "")
        elif isinstance(dep_obj, list):
            dependencies = dep_obj

        lookup[sec_num] = {
            "dependencies": dependencies,
            "reasoning": reasoning
        }

        if section.get("subsections"):
            build_lookup(section["subsections"], lookup)

    return lookup

lookup = build_lookup(data["sections"])

# ── 3. Read input CSV + enrich ───────────────────────────────────────────────

input_file = "section dependency.csv"   # your existing CSV
output_file = "section_dependencies.csv"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    fieldnames = reader.fieldnames + ["section_dependency", "reasoning"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        sec_num = row.get("section_number", "").strip()

        json_data = lookup.get(sec_num, {})

        deps = json_data.get("dependencies", [])
        reasoning = json_data.get("reasoning", "")

        # Format
        row["section_dependency"] = ", ".join(deps) if deps else ""
        row["reasoning"] = reasoning.replace("\n", " ").strip()

        writer.writerow(row)

print(f"CSV file generated: {output_file}")