import os
import sys
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# We use exception to fall back gracefully if run before requirements installed
from sentence_transformers import SentenceTransformer

def main():
    threads_path = os.path.join(project_root, "data", "amazon_help_threads.json")
    if not os.path.exists(threads_path):
        print(f"Data missing: {threads_path}")
        return
        
    print(f"Loading {threads_path}...")
    with open(threads_path, 'r', encoding='utf-8') as f:
        threads = json.load(f)
        
    # Exclude the golden set (eval set)
    import pandas as pd
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df_golden = pd.read_csv(csv_path)
    golden_thread_ids = set(df_golden['thread_id'].astype(str))
        
    resolved = [t for t in threads if t.get('is_resolved') and str(t['thread_id']) not in golden_thread_ids]
    print(f"Loaded {len(threads)} total threads. Filtered to {len(resolved)} resolved threads (excluding golden set).")
    
    metadata = []
    texts = []
    
    for t in resolved:
        msgs = t['messages']
        # Find first customer message
        first_cust = next((m['text'] for m in msgs if m['role'] == 'customer'), None)
        # Find final resolving reply (last message from brand)
        final_brand = next((m['text'] for m in reversed(msgs) if m['role'] == 'brand'), None)
        
        # Super strict safety filtering
        if first_cust and final_brand and len(first_cust) > 5 and len(final_brand) > 5:
            texts.append(first_cust)
            metadata.append({
                "thread_id": t['thread_id'],
                "customer_text": first_cust,
                "brand_resolution": final_brand
            })
            
    print(f"Extracted {len(metadata)} valid resolved thread pairs for indexing.")
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding texts... This may take a moment.")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    index_dir = os.path.join(project_root, "data", "resolved_threads_index")
    os.makedirs(index_dir, exist_ok=True)
    
    emb_path = os.path.join(index_dir, "embeddings.npy")
    mt_path = os.path.join(index_dir, "metadata.json")
    
    np.save(emb_path, embeddings)
    with open(mt_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Successfully indexed and saved {len(metadata)} resolved threads to {index_dir}")

if __name__ == "__main__":
    main()
