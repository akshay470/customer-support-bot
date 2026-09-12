import os
import sys
import pandas as pd

# Avoid path issues by dynamically finding the root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.baseline_simple import classify_keyword

def calc_stats_from_df(df, label_col='intent_label', pred_col='predicted'):
    total = len(df)
    correct = (df[pred_col] == df[label_col]).sum()
    overall = correct / total if total > 0 else 0
    intent_acc = {}
    for intent in df[label_col].unique():
        sub = df[df[label_col] == intent]
        c = (sub[pred_col] == intent).sum()
        intent_acc[intent] = c / len(sub) if len(sub) > 0 else 0
    return overall, intent_acc, total

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path)
    label_col = 'intent_label'
    df = df[df[label_col].notna()]
    
    # 1. Trivial Baseline
    majority_class = df[label_col].mode()[0]
    df_trivial = df.copy()
    df_trivial['predicted'] = majority_class
    
    # 2. Simple Baseline
    df_simple = df.copy()
    df_simple['predicted'] = df_simple['text'].apply(classify_keyword)
    
    # 3. LLM Baselines
    df_zero = pd.read_csv(os.path.join(project_root, "eval", "classifier_eval_results.csv"))
    df_few = pd.read_csv(os.path.join(project_root, "eval", "classifier_eval_results_fewshot.csv"))
    
    o_triv, i_triv, t_triv = calc_stats_from_df(df_trivial, 'intent_label', 'predicted')
    o_simp, i_simp, t_simp = calc_stats_from_df(df_simple, 'intent_label', 'predicted')
    
    # Zero/Few use 'actual' instead of intent_label in eval results
    o_zero, i_zero, t_zero = calc_stats_from_df(df_zero, 'actual', 'predicted')
    o_few, i_few, t_few = calc_stats_from_df(df_few, 'actual', 'predicted')
    
    all_intents = sorted(list(set(i_triv.keys()) | set(i_simp.keys()) | set(i_zero.keys()) | set(i_few.keys())))
    
    print("\n# BASELINE COMPARISON OVERVIEW")
    print(f"| Metric | Trivial (n={t_triv}) | Simple (n={t_simp}) | Zero-Shot LLM (n={t_zero}) | Few-Shot LLM (n={t_few}) |")
    print("|---|---|---|---|---|")
    print(f"| **Overall Accuracy** | {o_triv*100:.1f}% | {o_simp*100:.1f}% | {o_zero*100:.1f}% | {o_few*100:.1f}% |")
    
    print("\n# PER-INTENT BREAKDOWN")
    print(f"| Intent | Trivial | Simple | Zero-Shot | Few-Shot |")
    print("|---|---|---|---|---|")
    
    for intent in all_intents:
        def fmt(dct):
            return f"{dct[intent]*100:.1f}%" if intent in dct else "N/A"
        print(f"| {intent} | {fmt(i_triv)} | {fmt(i_simp)} | {fmt(i_zero)} | {fmt(i_few)} |")

if __name__ == "__main__":
    main()
