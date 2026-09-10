"""
sample_for_labeling.py

Extracts the first customer message from each thread and randomly samples 150 
stratified roughly by date, outputting to a CSV for manual intent labeling.
"""

import os
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_path = os.path.join("data", "amazon_help_threads.json")
    output_path = os.path.join("notebooks", "sample_for_labeling.csv")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}. Please run main.py first.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        threads = json.load(f)
        
    first_messages = []
    
    for thread in threads:
        # Find the first customer message in the thread
        for msg in thread.get('messages', []):
            if msg.get('role') == 'customer':
                first_messages.append({
                    "thread_id": thread.get("thread_id"),
                    "tweet_id": msg.get("tweet_id", thread.get("thread_id")),
                    "text": msg.get("text"),
                    "timestamp": msg.get("timestamp")
                })
                break # Only take the very first customer message
                
    total_first_msgs = len(first_messages)
    logger.info(f"Total first messages extracted: {total_first_msgs}")
    
    if total_first_msgs == 0:
        logger.warning("No customer messages found to sample.")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(first_messages)
    # Ensure timestamps are parsed properly despite possible varying formats
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    if df.empty:
        logger.warning("No valid timestamps found.")
        return

    # Stratify roughly by day
    df['date'] = df['timestamp'].dt.date
    
    sample_size = min(150, len(df))
    
    try:
        # Proportionate stratified sampling by date
        sampled_df = df.groupby('date', group_keys=False).apply(
            lambda x: x.sample(frac=sample_size/len(df), random_state=42)
        )
        
        # Because of rounding, we might end up with slightly more or fewer than 150 rows.
        # If fewer, randomly sample from the remaining original rows.
        if len(sampled_df) < sample_size:
            remaining = df.drop(sampled_df.index).sample(n=sample_size - len(sampled_df), random_state=42)
            sampled_df = pd.concat([sampled_df, remaining])
        # If more, simply truncate via random sample
        elif len(sampled_df) > sample_size:
            sampled_df = sampled_df.sample(n=sample_size, random_state=42)
            
    except ValueError:
        # Fallback to pure random sample
        sampled_df = df.sample(n=sample_size, random_state=42)
        
    # Drop our temporary date column
    sampled_df = sampled_df.drop(columns=['date'])
    
    # Add empty intent_label column for manual curation
    sampled_df['intent_label'] = ""
    
    # Save the output CSV
    os.makedirs("notebooks", exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    
    dates = sampled_df['timestamp'].dt.strftime('%Y-%m-%d')
    date_range = f"{dates.min()} to {dates.max()}"
    
    print("\n" + "="*50)
    print("SAMPLING LOG")
    print("="*50)
    print(f"Total Customer First-Messages: {total_first_msgs}")
    print(f"Messages Sampled:              {len(sampled_df)}")
    print(f"Date Range of Sample:          {date_range}")
    print(f"Output saved to:               {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
