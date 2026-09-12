import os
import sys
import json
import logging
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ingest import filter_brand_data
from src.threads import reconstruct_threads
from src.clean import clean_threads

logging.basicConfig(level=logging.INFO)

def main():
    input_path = os.path.join(project_root, "data", "twcs.csv")
    output_filtered = os.path.join(project_root, "data", "amazon_help_filtered.csv")
    output_threads = os.path.join(project_root, "data", "amazon_help_threads.json")
    
    # 1. Ingest & Filter
    df = filter_brand_data(input_path, output_filtered, brand_handle="AmazonHelp")
    if df is None:
        raise ValueError("Failed to ingest twcs.csv")
        
    # 2. Reconstruct Threads
    threads = reconstruct_threads(df, brand_handle="AmazonHelp")
    
    # 3. Clean Threads
    cleaned = clean_threads(threads)
    
    with open(output_threads, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=4)
        
    print(f"Data generation complete! Saved {len(cleaned)} threads to {output_threads}")
    
if __name__ == "__main__":
    main()
