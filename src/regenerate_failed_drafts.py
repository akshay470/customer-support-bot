import os
import sys
import pandas as pd
import time
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.classify import classify_intent
from src.draft_reply import draft_reply, load_index

def main():
    eval_dir = os.path.join(project_root, "eval")
    csv_path = os.path.join(eval_dir, "human_review_template.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Identify failed rows
    failed_mask = df['drafted_reply'].str.contains("Draft generation failed", na=False)
    failed_indices = df[failed_mask].index.tolist()
    
    if not failed_indices:
        print("No failed drafts found. All rows already have real content.")
        return
        
    print(f"Found {len(failed_indices)} failed drafts. Beginning regeneration...")
    
    load_index()
    
    success_count = 0
    fail_count = 0
    
    for idx in failed_indices:
        print(f"Regenerating row index {idx}...")
        customer_text = df.at[idx, 'customer_text']
        
        # We need intent to pass to draft_reply.
        try:
            classification_result = classify_intent(customer_text)
            intent = "other"
            
            if isinstance(classification_result, dict):
                intent = classification_result.get("intent", "other")
            else:
                try:
                    parsed = json.loads(classification_result)
                    intent = parsed.get("intent", "other")
                except:
                    classification_str = str(classification_result).lower()
                    for possible_intent in ["account_access", "account_settings", "agent_escalation", "contact_request", "delivery_delay", "order_status", "product_availability", "wrong_delivery", "other"]:
                        if possible_intent in classification_str:
                            intent = possible_intent
                            break
        except Exception as e:
            print(f"  -> Fast-failing classification due to {e}")
            intent = "other"
                            
        print(f"  -> Intent classified as: {intent}")
        time.sleep(5.0) # Requested 5+ delay
        
        reply_result = draft_reply(customer_text, intent)
        drafted_text = reply_result.get("draft_reply", "")
        
        if drafted_text and not reply_result.get("failed"):
            df.at[idx, 'drafted_reply'] = drafted_text
            print(f"  -> Success.")
            success_count += 1
        else:
            print(f"  -> Still failed.")
            fail_count += 1
            
        time.sleep(5.0) # Requested delay between calls
        
        # Removed continuous save to avoid permission errors on Windows
    
    print("\n--- Regeneration Complete ---")
    print(f"Total Attempted: {len(failed_indices)}")
    print(f"Successfully Regenerated: {success_count}")
    print(f"Still Failed: {fail_count}")
    
    try:
        df.to_csv(csv_path, index=False)
        print(f"Saved {success_count} updates to {csv_path}")
    except PermissionError:
        backup = csv_path.replace(".csv", "_backup.csv")
        df.to_csv(backup, index=False)
        print(f"File was locked. Saved to fallback: {backup}")

if __name__ == "__main__":
    main()
