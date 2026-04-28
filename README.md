# Running the Pipeline

You can run the full pipeline or individual modules using the central orchestrator script. 

## Interactive Mode (Recommended)
Run the script with no arguments. It will prompt you to enter the module numbers (comma-separated) and ask for the start/end rows if `Batch Transform` is selected:

```bash
python src/run_pipeline.py
```

## CLI Mode
You can bypass the interactive menu by specifying arguments directly:

```bash
# Run all stages (1 to 3)
python src/run_pipeline.py --from 1 --until 3

# Run only a specific stage (e.g., Stage 2: Batch Transform)
python src/run_pipeline.py --only 2

# Run stages 2 and 3
python src/run_pipeline.py --from 2 --until 3
```

Any extra arguments provided will be passed directly to the underlying scripts:
```bash
# Run only batch transform (stage 2) with specific start/end and authoring type
python src/run_pipeline.py --only 2 --start 1 --end 31 --authoring_type csr
```
