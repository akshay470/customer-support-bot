import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

def main():
    judge_path = 'eval/judge_scores.csv'
    human_path = 'eval/human_review_template.csv'
    
    if not os.path.exists(judge_path) or not os.path.exists(human_path):
        print("Missing one of the CSV files.")
        return
        
    df_judge = pd.read_csv(judge_path)
    df_human = pd.read_csv(human_path)
    
    # Merge on customer_text
    df = pd.merge(df_judge, df_human, on='customer_text', how='inner')
    
    dimensions = ['groundedness', 'relevance', 'tone', 'completeness', 'overall']
    paired_cols = []
    
    for dim in dimensions:
        human_col = f'human_{dim}'
        paired_cols.append(dim)
        paired_cols.append(human_col)
        
        # Ensure numeric types to handle missing blanks
        if dim in df.columns:
            df[dim] = pd.to_numeric(df[dim], errors='coerce')
        if human_col in df.columns:
            df[human_col] = pd.to_numeric(df[human_col], errors='coerce')
            
    # Drop rows with NaN in any of the scored columns
    df_clean = df.dropna(subset=paired_cols).copy()
    num_compared = len(df_clean)
    
    if num_compared == 0:
        print("0 rows available for comparison after dropping blanks.")
        return
        
    correlations = []
    
    print(f"\n--- Judge vs Human Agreement Analysis ({num_compared} rows compared) ---\n")
    print(f"{'Dimension':<15} | {'Exact Match %':<15} | {'Mean Abs Diff':<15} | {'Correlation (Spearman)':<25}")
    print("-" * 75)
    
    for dim in dimensions:
        human_col = f'human_{dim}'
        judge_scores = df_clean[dim]
        human_scores = df_clean[human_col]
        
        exact_match = (judge_scores == human_scores).mean() * 100
        mean_abs_diff = (judge_scores - human_scores).abs().mean()
        
        if judge_scores.nunique() > 1 and human_scores.nunique() > 1:
            corr, _ = spearmanr(judge_scores, human_scores)
        else:
            corr = float('nan')
            
        if not np.isnan(corr):
            correlations.append(corr)
            corr_str = f"{corr:.3f}"
        else:
            corr_str = "NaN (Zero Var)"
            
        print(f"{dim.capitalize():<15} | {exact_match:>13.1f}% | {mean_abs_diff:>13.2f} | {corr_str:>22}")
        
    print("-" * 75)
    
    # Summary sentence
    avg_corr = np.nanmean(correlations) if correlations else 0
    if avg_corr >= 0.7:
        strength = "strong"
    elif avg_corr >= 0.4:
        strength = "moderate"
    elif avg_corr > 0:
        strength = "weak"
    else:
        strength = "poor or practically none"
        
    print(f"\nCompared {num_compared} complete rows. Agreement is generally {strength} (Average Spearman correlation: {avg_corr:.2f}).")
    
    # Save the output
    out_cols = ['customer_text', 'drafted_reply_x'] + paired_cols
    df_out = df_clean[out_cols].copy()
    df_out = df_out.rename(columns={'drafted_reply_x': 'drafted_reply'})
    
    out_path = 'eval/judge_human_agreement.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\nFull row-by-row comparison saved to: {out_path}")

if __name__ == "__main__":
    main()
