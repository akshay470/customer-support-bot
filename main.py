"""
main.py

End-to-end execution of the data pipeline.
Sets up the dataset ingestion, chains threads, cleans data, and reports basic statistics.
"""

import os
import json
import logging
from src.ingest import filter_brand_data
from src.threads import reconstruct_threads
from src.clean import clean_threads
from datetime import datetime
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AI Customer Support Agent Data Pipeline")
    
    # Define paths
    data_dir = "data"
    raw_data_path = os.path.join(data_dir, "twcs.csv")
    filtered_data_path = os.path.join(data_dir, "amazon_help_filtered.csv")
    threads_output_path = os.path.join(data_dir, "amazon_help_threads.json")
    
    # Check if raw data exists
    if not os.path.exists(raw_data_path):
        logger.error(f"FATAL: Raw data file not found at {raw_data_path}. Please download the 'Customer Support on Twitter' dataset from Kaggle and place 'twcs.csv' in the data/ directory, per the README instructions.")
        raise FileNotFoundError(f"Missing required dataset: {raw_data_path}")
    
    # Step 1: Ingest
    logger.info("--- Step 1: Ingesting & Filtering Data ---")
    filtered_df = filter_brand_data(
        input_path=raw_data_path, 
        output_path=filtered_data_path,
        brand_handle="AmazonHelp"
    )
    if filtered_df is None or filtered_df.empty:
        logger.error("No data extracted. Exiting.")
        return
        
    # Step 2: Reconstruct Threads
    logger.info("--- Step 2: Reconstructing Threads ---")
    raw_threads = reconstruct_threads(filtered_df, brand_handle="AmazonHelp")
    
    # Step 3: Clean and Format
    logger.info("--- Step 3: Cleaning Text & Formatting ---")
    final_threads = clean_threads(raw_threads)
    
    # Save final threads to json
    with open(threads_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_threads, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved finalized threads to {threads_output_path}")
    
    # Generate Stats
    if not final_threads:
        logger.warning("No valid threads found after cleaning.")
        return

    total_threads = len(final_threads)
    avg_length = sum(len(t['messages']) for t in final_threads) / total_threads
    
    all_timestamps = []
    for t in final_threads:
        for m in t['messages']:
            all_timestamps.append(m['timestamp'])
    
    date_range = (min(all_timestamps), max(all_timestamps)) if all_timestamps else ("N/A", "N/A")
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Total Threads Created: {total_threads}")
    print(f"Average Thread Length: {avg_length:.1f} messages")
    print(f"Date Range of Data:    {date_range[0]} to {date_range[1]}")
    print("="*50)
    
    # Print sample of up to 5 threads
    sample_size = min(5, total_threads)
    print(f"\nSample of Output Threads (First {sample_size}):")
    print(json.dumps(final_threads[:sample_size], indent=2))
    print("="*50)

if __name__ == "__main__":
    main()
