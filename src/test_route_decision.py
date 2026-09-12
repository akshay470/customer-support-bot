import os
import sys
import pandas as pd
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.classify import classify_intent
from src.draft_reply import draft_reply, load_index
from src.route_decision import route_decision

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path).dropna(subset=['intent_label'])
    
    # Take 10 diverse samples
    samples = df.sample(n=10, random_state=42)
    
    # Preload the embeddings index so it doesn't log during the table output
    load_index()
    
    print("\n" + "="*80)
    print("=== END-TO-END PIPELINE: ROUTING DECISION TEST ===")
    print("="*80)
    
    for idx, row in samples.iterrows():
        customer_text = row['text'].encode('ascii', 'ignore').decode('ascii')
        
        # 1. Classify
        classification_result = classify_intent(customer_text)
        
        if isinstance(classification_result, dict):
            intent = classification_result.get("intent", "other")
            confidence = classification_result.get("confidence", "low")
        else:
            try:
                parsed = json.loads(classification_result)
                intent = parsed.get("intent", "other")
                confidence = parsed.get("confidence", "low")
            except:
                intent_fallback = "other"
                classification_str = str(classification_result).lower()
                for possible_intent in ["account_access", "account_settings", "agent_escalation", "contact_request", "delivery_delay", "order_status", "product_availability", "wrong_delivery", "other"]:
                    if possible_intent in classification_str:
                        intent_fallback = possible_intent
                        break
                intent = intent_fallback
                confidence = "low" if "low" in classification_str else ("high" if "high" in classification_str else "medium")
            
        # 2. Draft Reply & Retrieve examples
        # We pass intent instead of true intent_label to simulate real-world e2e
        reply_result = draft_reply(customer_text, intent)
        scores = reply_result.get("retrieval_similarity_scores", [])
        
        # 3. Decision Engine
        route_out = route_decision(
            customer_message=customer_text,
            intent=intent,
            confidence=confidence,
            draft_reply=reply_result,
            retrieval_similarity_scores=scores
        )
        
        print(f"\nCUSTOMER: {customer_text}")
        print(f"INTENT: {intent} (Confidence: {confidence})")
        print(f"DECISION: {route_out['decision']}")
        print(f"REASON:   {route_out['reason']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
