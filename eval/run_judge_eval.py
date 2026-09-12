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
                
        reply_result = draft_reply(customer_text, intent)
        drafted_text = reply_result.get("draft_reply", "")
        context_str = reply_result.get("context_used", "None")
        
        # 2. Judge evaluation
        scores = judge_reply(
            customer_message=customer_text,
            drafted_reply=drafted_text,
            retrieved_context=[context_str]
        )
        
        print(f"Judged [{idx}]: {intent} -> Overall {scores.get('overall', 0)}/5")
        
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
    
    # Calculate distributions
    print("\n" + "="*40)
    print("AVERAGE JUDGE SCORES (1-5)")
    print("="*40)
    print(f"Groundedness: {df_results['groundedness'].mean():.2f}")
    print(f"Relevance:    {df_results['relevance'].mean():.2f}")
    print(f"Tone:         {df_results['tone'].mean():.2f}")
    print(f"Completeness: {df_results['completeness'].mean():.2f}")
    print(f"OVERALL:      {df_results['overall'].mean():.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
