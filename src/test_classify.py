"""
test_classify.py
Test script for the intentions classification module.
"""
import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.classify import classify_intent, get_client

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    
    df = pd.read_csv(csv_path)
    
    # We want 10 samples with diverse values
    label_col = 'intent_label'
    if df[label_col].isna().all():
        label_col = 'llm_suggested_label'
        
    labeled_df = df[df[label_col].notna()]
    
    if len(labeled_df) == 0:
        print("Error: No labeled rows found in the CSV!")
        return
        
    samples = labeled_df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    
    if len(samples) < 10:
        remaining = 10 - len(samples)
        extras = labeled_df[~labeled_df.index.isin(samples.index)].sample(n=remaining, random_state=42)
        samples = pd.concat([samples, extras], ignore_index=True)
        
    samples = samples.head(10) # ensure strictly 10
        
    client = get_client()
    
    print(f"{'TEXT':<55} | {'PREDICTED':<20} | {'ACTUAL':<20} | {'MATCH':<5} | {'CONFIDENCE'}")
    print("-" * 130)
    
    for idx, row in samples.iterrows():
        text = str(row['text'])
        # Truncate text for printing safely without breaking lines
        short_text = text.replace('\n', ' ').encode('ascii', 'ignore').decode('ascii')
        short_text = short_text[:52] + "..." if len(short_text) > 52 else short_text.ljust(55)
        
        actual = str(row[label_col])
        
        result = classify_intent(text, client=client)
        predicted = result["intent"]
        confidence = result["confidence"]
        
        match = "YES" if predicted == actual else "NO"
        
        print(f"{short_text:<55} | {predicted:<20} | {actual:<20} | {match:<5} | {confidence}")

if __name__ == "__main__":
    main()
