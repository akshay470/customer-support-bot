"""
clean.py

Text cleaning and PII stripping for conversational threads.
"""

import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Strips URLs, masks @mentions (except for stripping it completely when too repetitive), 
    and removes excessive whitespace while preserving the core complaint/question text.
    """
    if not isinstance(text, str):
        return ""
        
    # Replace basic HTML entities often found in raw text
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    
    # Strip URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Mask @mentions/handles as [USER]
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Remove excessive whitespace (newlines, tabs, multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_threads(threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies text cleaning to all messages within the threads and drops duplicates/spam.
    
    Args:
        threads (List[Dict[str, Any]]): Uncleaned thread records.
        
    Returns:
        List[Dict[str, Any]]: Cleaned thread records.
    """
    logger.info("Cleaning thread texts...")
    cleaned_threads = []
    
    for thread in threads:
        cleaned_messages = []
        is_spam = False
        
        for msg in thread['messages']:
            original_text = msg['text']
            cleaned = clean_text(original_text)
            
            # Super basic empty message filter
            if len(cleaned.strip()) < 2:
                continue
                
            # Naive spam check (e.g. repeated same characters like 'aaaaa')
            if re.match(r'^(.)\1{10,}$', cleaned):
                is_spam = True
                break
                
            cleaned_messages.append({
                "tweet_id": msg.get("tweet_id"),
                "author": msg["author"],
                "role": msg["role"],
                "text": cleaned,
                "timestamp": msg["timestamp"]
            })
            
        # Only keep threads that are not spam and still have >= 2 messages after cleaning
        if not is_spam and len(cleaned_messages) > 1:
            cleaned_threads.append({
                "thread_id": thread["thread_id"],
                "messages": cleaned_messages,
                "is_resolved": thread["is_resolved"]
            })
            
    logger.info(f"Remaining active threads after cleaning: {len(cleaned_threads)}")
    return cleaned_threads
