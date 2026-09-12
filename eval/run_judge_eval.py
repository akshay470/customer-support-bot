import os
import sys
import pandas as pd
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.classify import classify_intent
from src.draft_reply import draft_reply, load_index
from src.judge_reply import judge_reply

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path).dropna(subset=['intent_label'])
    
    # 30 diverse samples for judge evaluation
    samples = df.sample(n=30, random_state=42).copy()
    
    load_index()
    
    results = []
    
    print(f"Running LLM-as-judge evaluation on {len(samples)} samples...\n")
    
    for idx, row in samples.iterrows():
        customer_text = row['text'].encode('ascii', 'ignore').decode('ascii')
        
        # 1. Pipeline execution
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
                
        time.sleep(3.0) # Buffer before draft
        
        reply_result = draft_reply(customer_text, intent)
        drafted_text = reply_result.get("draft_reply", "")
        if not drafted_text or reply_result.get("failed"):
            print(f"Skipped [{idx}]: draft generation failed")
            continue
            
        context_str = reply_result.get("context_used", "None")
        
        time.sleep(3.0) # Buffer before judge
        
        # 2. Judge evaluation
        scores = judge_reply(
            customer_message=customer_text,
            drafted_reply=drafted_text,
            retrieved_context=[context_str]
        )
        if not isinstance(scores, dict):
            scores = {}
            
        overall = scores.get('overall')
        overall_str = overall if overall is not None else "failed"
        print(f"Judged [{idx}]: {intent} -> Overall {overall_str}/5")
        
        record = {
            "customer_text": customer_text,
            "intent": intent,
            "drafted_reply": drafted_text,
            "groundedness": scores.get("groundedness"),
            "relevance": scores.get("relevance"),
            "tone": scores.get("tone"),
            "completeness": scores.get("completeness"),
            "overall": scores.get("overall"),
            "judge_reasoning": scores.get("judge_reasoning", "")
        }
        results.append(record)
        
        # Add a sleep to prevent Groq rate limits
        time.sleep(2.0)
    
    eval_dir = os.path.join(project_root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save Full Judged Output
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        out_path = os.path.join(eval_dir, "judge_scores.csv")
        df_results.to_csv(out_path, index=False)
        print(f"\nSaved judge scores to {out_path}")
        
        # Save Template for Human Review
        human_cols = [
            "customer_text", "drafted_reply",
            "human_groundedness", "human_relevance", "human_tone", 
            "human_completeness", "human_overall", "human_reasoning"
        ]
        df_human = df_results[["customer_text", "drafted_reply"]].copy()
        for col in human_cols[2:]:
            df_human[col] = ""
            
        human_path = os.path.join(eval_dir, "human_review_template.csv")
        df_human.to_csv(human_path, index=False)
        print(f"Saved human review template to {human_path}")
    else:
        print("\nNo rows successfully evaluated to save.")
    
    # Calculate distributions
    if not df_results.empty and 'overall' in df_results.columns:
        valid_scores = df_results.dropna(subset=['overall'])
    else:
        valid_scores = pd.DataFrame()
        
    num_valid = len(valid_scores)
    
    print("\n" + "="*40)
    print(f"AVERAGE JUDGE SCORES ({num_valid}/30 Valid)")
    print("="*40)
    if num_valid > 0:
        print(f"Groundedness: {valid_scores['groundedness'].mean():.2f}")
        print(f"Relevance:    {valid_scores['relevance'].mean():.2f}")
        print(f"Tone:         {valid_scores['tone'].mean():.2f}")
        print(f"Completeness: {valid_scores['completeness'].mean():.2f}")
        print(f"OVERALL:      {valid_scores['overall'].mean():.2f}")
    else:
        print("No valid scores to calculate averages.")
    print("="*40)

if __name__ == "__main__":
    main()
