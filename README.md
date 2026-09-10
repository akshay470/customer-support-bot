# AI Customer Support Agent

## Setup
(Placeholder: Setup instructions will be added here)

## Data Pipeline
To run the data pipeline end-to-end:
```bash
python main.py
```

## Manual Labeling
Before building an intent classifier, generate a random stratified sample of 150 initial customer messages by running:
```bash
python src/sample_for_labeling.py
```
This script will produce `notebooks/sample_for_labeling.csv`. Use this file to define a taxonomy and manually hand-label intents in the `intent_label` column.
