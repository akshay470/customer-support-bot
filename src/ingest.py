"""
ingest.py

Loads raw CSV data and filters for a specific brand's conversations.
"""

import pandas as pd
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_brand_data(input_path: str, output_path: str, brand_handle: str = "AmazonHelp") -> Optional[pd.DataFrame]:
    """
    Loads the raw CSV dataset and filters to only tweets where the brand was the author 
    OR tweets that are part of a thread involving the brand.
    
    Args:
        input_path (str): Path to raw CSV file.
        output_path (str): Path to save the filtered data (CSV or Parquet).
        brand_handle (str): Brand's Twitter handle to filter on.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    logger.info(f"Loading raw data from {input_path}")
    try:
        # Load the raw dataset
        # Note: We enforce string types for IDs to preserve large numeric strings accurately.
        df = pd.read_csv(input_path, dtype={"tweet_id": str, "in_response_to_tweet_id": str, "author_id": str})
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return None

    logger.info(f"Original dataset size: {len(df)} rows")
    
    # 1. Identify all tweets authored by the brand
    brand_tweets = df[df['author_id'] == brand_handle]
    brand_tweet_ids = set(brand_tweets['tweet_id'].dropna())
    
    # 2. Identify tweets that the brand responded to
    # (The brand's tweets have an in_response_to_tweet_id which points to the customer's tweet_id)
    responded_to_by_brand_ids = set(brand_tweets['in_response_to_tweet_id'].dropna())
    
    # 3. Identify tweets that replied to the brand
    # (Customer tweets that have in_response_to_tweet_id in the brand's tweet IDs)
    replies_to_brand = df[df['in_response_to_tweet_id'].isin(brand_tweet_ids)]
    replies_to_brand_ids = set(replies_to_brand['tweet_id'].dropna())
    
    # Combine all relevant tweet IDs
    relevant_ids = brand_tweet_ids.union(responded_to_by_brand_ids).union(replies_to_brand_ids)
    
    # Filter the main DataFrame for these relevant IDs
    filtered_df = df[df['tweet_id'].isin(relevant_ids)].copy()
    
    logger.info(f"Filtered dataset size ({brand_handle} threads): {len(filtered_df)} rows")
    
    # Save output
    if output_path.endswith('.parquet'):
        filtered_df.to_parquet(output_path, index=False)
    else:
        filtered_df.to_csv(output_path, index=False)
        
    logger.info(f"Saved filtered data to {output_path}")
    
    return filtered_df
