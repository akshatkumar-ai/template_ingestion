## batch_transform.py

Processes section instructions from a JSON file (`section_instructions.json`) one section per LLM call.
Uses `system_prompt.txt` as the prompt template. Saves per-section prompts and responses to `outputs/<section_number>/`.

```bash
# Protocol authoring (default)
python batch_transform.py --start 1 --end 31

# CSR authoring
python batch_transform.py --authoring_type csr --start 1 --end 31

# SAP authoring
python batch_transform.py --authoring_type sap --start 1 --end 31

# With metric tracking (streaming + per-section time/token/stop_reason reporting)
python batch_transform.py --authoring_type csr --json csr_json_input.json --metrics
```

## experiment_transform.py

Sends all sections in a single bulk LLM call (streaming) using `system_prompt_complete_json.txt`.
Used to test time and token limits of the model when processing the entire document at once.
Saves the final resolved prompt to `outputs/experiment_prompt.txt` and raw response to `outputs/experiment_response_raw.txt`.

```bash
python experiment_transform.py --authoring_type csr --output experiment_output.json
```
