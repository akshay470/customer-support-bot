import os
import sys
import time
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.classify import classify_intent, get_client

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path)
    
    label_col = 'intent_label'
    if label_col not in df.columns or df[label_col].isna().all():
        print(f"Error: {label_col} column is missing or entirely null!")
        return
        
    df = df[df[label_col].notna()]
    
    if len(df) == 0:
        print("Error: No labeled rows found.")
        return
        
    client = get_client()
    
    results = []
    
    correct = 0
    total = len(df)
    
    intent_stats = {}
    confusions = {}
    
    print(f"Starting evaluation on {total} rows...")
    
    for i, (idx, row) in enumerate(df.iterrows()):
        text = str(row['text'])
        actual = str(row[label_col])
        
        start_time = time.time()
        
        result_dict = classify_intent(text, client=client)
        predicted = result_dict["intent"]
        confidence = result_dict["confidence"]
        
        match = "YES" if predicted == actual else "NO"
        
        results.append({
            "text": text,
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence,
            "match": match
        })
        
        if actual not in intent_stats:
            intent_stats[actual] = {'count': 0, 'correct': 0}
        intent_stats[actual]['count'] += 1
        
        if predicted == actual:
            correct += 1
            intent_stats[actual]['correct'] += 1
        else:
            pair = (actual, predicted)
            confusions[pair] = confusions.get(pair, 0) + 1
            
        elapsed = time.time() - start_time
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
            
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"Processed {i + 1} / {total} rows...")
            
    print("\n" + "="*50)
    print("OVERALL ACCURACY")
    print("="*50)
    print(f"{correct}/{total} ({(correct/total)*100:.1f}%)")
    
    print("\n" + "="*50)
    print("PER-INTENT BREAKDOWN")
    print("="*50)
    print(f"{'Intent':<25} | {'Count':<5} | {'Accuracy'}")
    print("-" * 50)
    for intent, stats in sorted(intent_stats.items()):
        cnt = stats['count']
        corr = stats['correct']
        acc = (corr / cnt) * 100 if cnt > 0 else 0
        print(f"{intent:<25} | {cnt:<5} | {acc:.1f}%")
        
    print("\n" + "="*50)
    print("CONFUSION SUMMARY (actual -> predicted)")
    print("="*50)
    if not confusions:
        print("No confusions found! 100% perfect.")
    else:
        # Sort by frequency descending
        for (act, pred), freq in sorted(confusions.items(), key=lambda x: x[1], reverse=True):
            print(f"{act} predicted as {pred}: {freq} times")
            
    # Save results
    eval_dir = os.path.join(project_root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    results_path = os.path.join(eval_dir, "classifier_eval_results.csv")
    
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nSaved full results to {results_path}")

if __name__ == "__main__":
    main()
