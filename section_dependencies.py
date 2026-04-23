import json
import os
import re


# ── Bedrock client ────────────────────────────────────────────────────────────

def get_bedrock_client():
    print("*" * 50)
    print("Connecting to AWS BEDROCK....")

    import boto3
    from dotenv import load_dotenv

    load_dotenv(".env")

    AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN     = os.getenv("AWS_SESSION_TOKEN")
    AWS_REGION            = "us-east-1"

    if not AWS_ACCESS_KEY_ID:
        raise ValueError("AWS_ACCESS_KEY_ID is missing in the environment variables.")
    if not AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS_SECRET_ACCESS_KEY is missing in the environment variables.")
    if not AWS_SESSION_TOKEN:
        raise ValueError("AWS_SESSION_TOKEN is missing in the environment variables.")

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
            config=boto3.session.Config(read_timeout=2000),
        )
        print("*" * 50)
        return client
    except Exception as e:
        print(e)
        raise ValueError("Error in loading BEDROCK client!")


def generate(prompt: str, model_id: str, client=None) -> str:
    from dotenv import load_dotenv
    load_dotenv(".env")

    if "claude" not in model_id.lower() and "anthropic" not in model_id.lower():
        raise ValueError(
            f"Unknown model_id '{model_id}'. Only Claude models are currently supported."
        )

    print(f"Generating response using {model_id}")

    if client is None:
        client = get_bedrock_client()

    try:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt.strip()}],
                }
            ],
        }
        response       = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]

    except Exception as e:
        print(e)
        raise ValueError("Error in generation!")


# ── 1. Load / Save ────────────────────────────────────────────────────────────

def load_protocol(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_protocol(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


# ── 2. TOC parser ─────────────────────────────────────────────────────────────

def parse_toc_from_markdown(path: str) -> list:
    """
    Parses a markdown TOC of the format:
        [1.3.1 Known Potential Risks   14](#known-potential-risks)

    Returns a flat list of dicts:
        [{"section_number": "1.3.1", "section_name": "Known Potential Risks"}, ...]
    """
    pattern = re.compile(
        r'^\[(\d[\d.]*)\s+(.+?)\s+\d+\]\(#[^)]*\)',
        re.MULTILINE
    )

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    toc = []
    for match in pattern.finditer(content):
        toc.append({
            "section_number": match.group(1).rstrip("."),
            "section_name":   match.group(2).strip()
        })

    if not toc:
        raise ValueError(
            f"No TOC entries found in {path}. "
            "Expected format: [1.2 Section Name   14](#anchor)"
        )

    print(f"  Parsed {len(toc)} entries from {path}")
    return toc


# ── 3. Section tree helpers ───────────────────────────────────────────────────

def get_section_sort_key(section_number: str) -> tuple:
    """Converts '5.1.2' → (5, 1, 2) for reliable numeric comparison."""
    try:
        return tuple(int(p) for p in section_number.split("."))
    except ValueError:
        return (float("inf"),)


def _flatten(sections: list, result: list):
    """Recursively flattens a nested section tree into a single list."""
    for s in sections:
        result.append({
            "section_number": s["section_number"],
            "section_name":   s["section_name"],
            "instructions":   s.get("instructions", "")
        })
        if s.get("subsections"):
            _flatten(s["subsections"], result)


def get_prior_context_sections(
    all_sections_flat: list,
    current_section_number: str,
    max_context_sections: int = 30
) -> list:
    """
    Returns up to max_context_sections sections that appear strictly before
    the current section. Always includes direct ancestors, then fills remaining
    slots with the closest preceding sections.
    """
    current_key = get_section_sort_key(current_section_number)

    # all sections strictly before current
    prior = [
        s for s in all_sections_flat
        if get_section_sort_key(s["section_number"]) < current_key
    ]

    # always include direct ancestors (e.g. 5, 5.3 for section 5.3.2)
    parts     = current_section_number.split(".")
    ancestors = {".".join(parts[:i]) for i in range(1, len(parts))}
    ancestor_sections = [s for s in prior if s["section_number"] in ancestors]

    # fill remaining slots with closest preceding sections (tail of prior)
    remaining_slots = max_context_sections - len(ancestor_sections)
    tail = prior[-remaining_slots:] if remaining_slots > 0 else []

    # merge, deduplicate, preserve order
    seen   = set()
    result = []
    for s in ancestor_sections + tail:
        if s["section_number"] not in seen:
            result.append(s)
            seen.add(s["section_number"])

    return result


# ── 4. Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(
    system_context: str,
    toc: list,
    section: dict,
    context_sections: list
) -> str:

    toc_text = "\n".join(
        f"  {item['section_number']}  {item['section_name']}"
        for item in toc
    )

    context_block = ""
    if context_sections:
        context_lines = [
            f"SECTION {s['section_number']} — {s['section_name']}:\n{s['instructions']}"
            for s in context_sections
        ]
        context_block = (
            "══════════════════════════════════════════════════════\n"
            "PRECEDING SECTIONS CONTEXT\n"
            "══════════════════════════════════════════════════════\n"
            "The following are the instructions for sections that precede the current "
            "section. Use these to identify true content dependencies — i.e. cases where "
            "the current section explicitly consumes a term, criterion, population, or "
            "procedure defined in one of these sections.\n\n"
            + "\n---\n".join(context_lines)
        )

    format_instruction = """\
══════════════════════════════════════════════════════
RESPONSE FORMAT
══════════════════════════════════════════════════════

Return ONLY a single valid JSON object. No text, explanation, or markdown before or after it.

{
  "dependencies": ["3.1", "5.1.1"],
  "reasoning": "One or two sentences justifying each included dependency."
}

If no dependencies exist:

{
  "dependencies": [],
  "reasoning": "This section is foundational and does not require any prior section to be written first."
}"""

    return f"""{system_context}

{format_instruction}

---

TABLE OF CONTENTS:
{toc_text}

---

{context_block}

---

Now identify the dependencies for the following section:

SECTION NUMBER: {section["section_number"]}
SECTION NAME: {section["section_name"]}
INSTRUCTIONS / CONTENT DESCRIPTION:
{section["instructions"]}

Which sections that appear BEFORE this section does it directly depend on?"""


# ── 5. Dependency resolution ──────────────────────────────────────────────────

def filter_forward_dependencies(deps: list, current_section_number: str) -> list:
    """Removes any dependency whose section number is >= current section."""
    current_key = get_section_sort_key(current_section_number)
    filtered, removed = [], []

    for dep in deps:
        if get_section_sort_key(dep) < current_key:
            filtered.append(dep)
        else:
            removed.append(dep)

    if removed:
        print(f"    ⚠ Ordering filter removed forward refs: {removed}")

    return filtered


def get_dependencies(
    system_context: str,
    toc: list,
    section: dict,
    all_sections_flat: list,
    client,
    model_id: str,
    max_context_sections: int
) -> dict:

    context_sections = get_prior_context_sections(
        all_sections_flat,
        section["section_number"],
        max_context_sections=max_context_sections
    )

    prompt = build_prompt(system_context, toc, section, context_sections)
    raw    = generate(prompt=prompt, model_id=model_id, client=client)

    print(f"    RAW RESPONSE: {repr(raw[:300])}")

    raw = raw.strip()

    if not raw:
        raise ValueError("Model returned an empty response")

    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()

    if not raw.startswith("{"):
        raise ValueError(f"Response does not start with '{{'. Got: {repr(raw[:200])}")

    parsed = json.loads(raw)

    # validate against TOC
    valid_numbers = {item["section_number"] for item in toc}
    deps = [d for d in parsed.get("dependencies", []) if d in valid_numbers]

    # enforce ordering
    deps = filter_forward_dependencies(deps, section["section_number"])

    # enforce cap
    if len(deps) > 5:
        print(f"    ⚠ Cap applied: trimming {len(deps)} deps to 5")
        deps = deps[:5]

    return {
        "dependencies": deps,
        "reasoning":    parsed.get("reasoning", "").strip()
    }


# ── 6. Recursive tree walker ──────────────────────────────────────────────────

def _fill_recursive(
    system_context: str,
    sections: list,
    toc: list,
    all_sections_flat: list,
    client,
    model_id: str,
    max_context_sections: int
):
    for section in sections:
        print(f"  Processing {section['section_number']} — {section['section_name']}")

        try:
            result = get_dependencies(
                system_context, toc, section,
                all_sections_flat, client, model_id, max_context_sections
            )
            section["section_dependency"] = {
                "dependencies": result["dependencies"],
                "reasoning":    result["reasoning"]
            }
            print(f"    → {result['dependencies']}")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            section["section_dependency"] = {
                "dependencies": [],
                "reasoning":    f"ERROR: {e}"
            }

        if section.get("subsections"):
            _fill_recursive(
                system_context, section["subsections"], toc,
                all_sections_flat, client, model_id, max_context_sections
            )


# ── 7. Main pipeline ──────────────────────────────────────────────────────────

MODEL_ID = "arn:aws:bedrock:us-east-1:533267065792:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"


def run_pipeline(
    input_path: str,
    output_path: str,
    prompt_path: str,
    toc_path: str,
    model_id: str = MODEL_ID,
    max_context_sections: int = 30
):
    print("Loading protocol JSON...")
    protocol = load_protocol(input_path)
    sections = protocol["sections"]

    print(f"Loading system prompt from {prompt_path}...")
    system_context = load_prompt(prompt_path)

    print(f"Parsing TOC from {toc_path}...")
    toc = parse_toc_from_markdown(toc_path)

    print("Connecting to Bedrock...")
    client = get_bedrock_client()

    all_sections_flat = []
    _flatten(sections, all_sections_flat)
    print(f"  {len(all_sections_flat)} total sections found in JSON")

    print("Filling dependencies...")
    _fill_recursive(
        system_context=system_context,
        sections=sections,
        toc=toc,
        all_sections_flat=all_sections_flat,
        client=client,
        model_id=model_id,
        max_context_sections=max_context_sections
    )

    print(f"Saving JSON to {output_path}...")
    save_protocol(protocol, output_path)
    print(f"Done. Processed {len(all_sections_flat)} sections.")


if __name__ == "__main__":
    run_pipeline(
        input_path="protocol_template_hierarchy.json",
        output_path="protocol_template_with_dependencies.json",
        prompt_path="section_dependency_prompt.txt",
        toc_path="TOC.md",
        max_context_sections=30
    )