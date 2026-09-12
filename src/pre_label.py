"""
pre_label.py

Uses an LLM to pre-fill draft intent labels for the customer support message sample.
Expects GEMINI_API_KEY to be set in a .env file.
"""

import os
import time
import logging
import pandas as pd
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    logging.error("Please install the Gemini SDK: pip install google-genai python-dotenv")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

TAXONOMY = """
- delivery_delay: package late, not yet arrived, stuck in transit
- order_status: asking where an order is, order number references, general order inquiries
- agent_escalation: complaints about being ignored, hung up on, no response from support, wants a human
- account_access: login issues, verification problems, can't access account
- account_settings: unsubscribe requests, notification/email preferences
- product_availability: asking if a product is sold/available in a region
- contact_request: asking for an email/phone/contact method
- wrong_delivery: received wrong item or unordered package
- other: doesn't fit any category above, spam, off-topic, unclear
"""

SYSTEM_PROMPT = f"""You are a customer support intent classifier.
Categorize the user's message into exactly ONE of the following categories:
{TAXONOMY}

Your response must be ONLY the category name (e.g., "delivery_delay" or "other"). Do not include quotes, punctuation, or any other text.
"""

def call_llm_with_retry(client, text, row_idx, max_retries=3):
    """
    Calls the Gemini API with simple exponential backoff for rate limits.
    """
    for attempt in range(max_retries):
        try:
            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                input=text,
                system_instruction=SYSTEM_PROMPT
            )
            val = interaction.output_text.strip().lower()
            # Basic cleanup in case the LLM includes punctuation
            return val.replace("'", "").replace('"', '').replace('.', '')
            
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logger.warning(f"Row {row_idx} API error: {type(e).__name__}: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Row {row_idx} failed after {max_retries} attempts. Final error: {type(e).__name__}: {e}")
                return "other"

def main():
    # Calculate explicit path to .env file in the project directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    
    # Load env vars explicitly from that file FIRST
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(f"No GEMINI_API_KEY found. Looked for .env file exactly at: {env_path}")
        return
        
    logger.info(f"Loaded .env successfully from: {env_path}")
    
    client = genai.Client(api_key=api_key)
    
    input_path = os.path.join("notebooks", "sample_for_labeling.csv")
    output_path = os.path.join("notebooks", "sample_for_labeling_prelabeled.csv")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
        
    df = pd.read_csv(input_path)
    
    if "text" not in df.columns:
        logger.error("Could not find 'text' column in the CSV.")
        return
        
    total_rows = len(df)
    logger.info(f"Starting LLM pre-labeling for {total_rows} rows...")
    
    suggested_labels = []
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        label = call_llm_with_retry(client, text, row_idx=idx)
        suggested_labels.append(label)
        
        # Simple progress tracking
        count = idx + 1
        if count % 10 == 0 or count == total_rows:
            logger.info(f"Labeled {count}/{total_rows}")
            
        # Add safety buffer between calls to avoid per-minute rate limits, even with billing active
        time.sleep(1.5)
        
    # Write to a NEW column, leaving intent_label untouched
    df['llm_suggested_label'] = suggested_labels
    
    # Save the output
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved pre-labeled data to {output_path}")
    
    # Print summary distribution
    print("\n" + "="*50)
    print("PRE-LABELING SUMMARY (Distribution)")
    print("="*50)
    distribution = df['llm_suggested_label'].value_counts()
    for cat, count in distribution.items():
        print(f"{cat.ljust(25)} {count}")
    print("="*50)

if __name__ == "__main__":
    main()
