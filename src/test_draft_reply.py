import os
import sys
import pandas as pd
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.draft_reply import draft_reply, load_index

logging.basicConfig(level=logging.WARNING)

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path).dropna(subset=['intent_label'])
    
    # Get 5 diverse samples visually tracking intents
    samples = df.groupby('intent_label').head(1).head(5)
    
    print("Preloading index and models. This might take a second...")
    load_index()
    
    print("\n" + "="*50)
    print("=== DRAFT REPLY RAG TEST ===")
    
    for idx, row in samples.iterrows():
        customer_text = row['text'].encode('ascii', 'ignore').decode('ascii')
        intent = row['intent_label']
        
        result = draft_reply(customer_text, intent)
        
        print("\n" + "="*50)
        print(f"INTENT: {intent}")
        print(f"CUSTOMER: {customer_text}")
        print("-" * 50)
        
        threads = result['retrieved_examples']
        scores = result['retrieval_similarity_scores']
        
        print(f"RETRIEVED EXAMPLES (IDs & Scores):")
        for tid, score in zip(threads, scores):
            print(f"- {tid} (similarity: {score:.3f})")
            
        print("-" * 50)
        print("DRAFTED REPLY:")
        safe_reply = result['draft_reply'].encode('ascii', 'ignore').decode('ascii')
        print(safe_reply)

if __name__ == "__main__":
    main()
