import os
import pandas as pd
import json

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    
    df = pd.read_csv(csv_path)
    label_col = 'intent_label'
    
    # Filter only labeled
    df = df[df[label_col].notna()]
    
    # Sample 2 rows per intent
    few_shots = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(n=min(2, len(x)), random_state=42))
    
    # Remaining are test set
    test_set = df.drop(few_shots.index)
    
    # Generate few shots dict
    few_shots_dict = {}
    for idx, row in few_shots.iterrows():
        intent = row[label_col]
        if intent not in few_shots_dict:
            few_shots_dict[intent] = []
        few_shots_dict[intent].append(row['text'])
        
    # Write to a Python file to easily import in classify.py
    with open(os.path.join(project_root, "src", "few_shots.py"), "w", encoding="utf-8") as f:
        f.write('FEW_SHOTS = ')
        json.dump(few_shots_dict, f, indent=4, ensure_ascii=False)
        f.write('\n')
        
    print(f"Sampled {len(few_shots)} few shot examples.")
    print(f"Remaining {len(test_set)} for testing.")

    # Save test set for eval_classifier
    test_set.to_csv(os.path.join(project_root, "eval", "held_out_test_set.csv"), index=False)

if __name__ == "__main__":
    main()
