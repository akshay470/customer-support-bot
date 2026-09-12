import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path)
    label_col = 'intent_label'
    df = df[df[label_col].notna()]
    
    majority_class = df[label_col].mode()[0]
    df['predicted'] = majority_class
    
    total = len(df)
    correct = (df['predicted'] == df[label_col]).sum()
    
    print(f"Majority class predicted: {majority_class}\n")
    print(f"OVERALL ACCURACY: {correct}/{total} ({(correct/total)*100:.1f}%)\n")
    
    print(f"{'Intent':<25} | {'Count':<5} | {'Accuracy'}")
    print("-" * 50)
    
    # Make sure we print in a sorted order
    for intent in sorted(df[label_col].unique()):
        sub = df[df[label_col] == intent]
        c = (sub['predicted'] == intent).sum()
        print(f"{intent:<25} | {len(sub):<5} | {(c/len(sub))*100:.1f}%")

if __name__ == "__main__":
    main()
