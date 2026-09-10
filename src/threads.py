"""
threads.py

Reconstructs conversational threads from individual tweets.
"""

import pandas as pd
from typing import List, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def reconstruct_threads(df: pd.DataFrame, brand_handle: str = "AmazonHelp") -> List[Dict[str, Any]]:
    """
    Reconstructs full conversation threads using in_response_to_tweet_id to chain tweets.
    Returns ordered sequences (customer -> brand -> customer...).
    
    Args:
        df (pd.DataFrame): Filtered DataFrame containing brand conversations.
        brand_handle (str): The brand's handle.
        
    Returns:
        List[Dict[str, Any]]: A list of structured thread records.
    """
    logger.info("Reconstructing threads...")
    
    # Create a mapping for quick lookup by tweet_id
    tweet_dict = df.set_index('tweet_id').to_dict('index')
    all_tweet_ids = set(tweet_dict.keys())
    
    threads = []
    visited = set()
    
    # Step 1: build a child-lookup dict (parent_id -> list of child_ids)
    children = {}
    for t_id, data in tweet_dict.items():
        parent_id = str(data.get('in_response_to_tweet_id'))
        if parent_id != 'nan' and parent_id:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(t_id)

    # Step 2: identify roots. A root is a tweet without a parent present in our dataset.
    roots = []
    for t_id, data in tweet_dict.items():
        parent_id = str(data.get('in_response_to_tweet_id'))
        if parent_id == 'nan' or not parent_id or parent_id not in all_tweet_ids:
            roots.append(t_id)
            
    # Step 3: traverse from roots to build threads
    for root_id in tqdm(roots, desc="Building threads"):
        if root_id in visited:
            continue
            
        current_thread_msgs = []
        
        # Simple DFS to follow the chain. 
        # Support threads on twitter are mostly 1-on-1 chains.
        stack = [root_id]
        while stack:
            # Pop the latest inserted to follow one branch deep
            curr_id = stack.pop()
            if curr_id in visited:
                continue
            visited.add(curr_id)
            
            tweet_data = tweet_dict[curr_id]
            author = tweet_data.get('author_id')
            role = 'brand' if author == brand_handle else 'customer'
            
            current_thread_msgs.append({
                "tweet_id": curr_id,
                "author": author,
                "role": role,
                "text": str(tweet_data.get('text', '')),
                "timestamp": str(tweet_data.get('created_at', ''))
            })
            
            # Add child responses to stack to continue chain
            if curr_id in children:
                stack.extend(children[curr_id])
                
        # Only keep threads that have at least 2 messages
        if len(current_thread_msgs) > 1:
            # Sort messages chronologically by timestamp
            current_thread_msgs.sort(key=lambda x: x["timestamp"])
            
            # Check if resolved (last message is from the brand)
            is_resolved = current_thread_msgs[-1]['role'] == 'brand'
            
            threads.append({
                "thread_id": root_id,
                "messages": current_thread_msgs,
                "is_resolved": is_resolved
            })
            
    logger.info(f"Reconstructed {len(threads)} valid threads (with >= 2 messages).")
    return threads
