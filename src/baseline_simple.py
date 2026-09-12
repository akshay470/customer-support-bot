import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

RULES = {
    "account_access": ["log in", "login", "password", "locked", "block", "verification", "access", "suspend"],
    "account_settings": ["unsubscribe", "preference", "email setting", "notification"],
    "agent_escalation": ["lie", "worst", "pathetic", "useless", "terrible", "joke", "human", "call", "complaint", "refund", "poor service", "help!", "help me"],
    "contact_request": ["email", "phone", "contact", "support page", "message private"],
    "delivery_delay": ["late", "wait", "delay", "stuck", "tracking", "shipped", "arrive", "where", "haven't received", "not come", "time"],
    "order_status": ["order", "dispatch", "pending", "status", "cancel", "charge"],
    "product_availability": ["stock", "sold", "available", "release", "when will", "pre order", "pre-order"],
    "wrong_delivery": ["wrong", "unordered", "error", "different", "mistake", "instead", "missing"]
}

def classify_keyword(text):
    text_lower = str(text).lower()
    for intent, keywords in RULES.items():
        for kw in keywords:
            if kw in text_lower:
                return intent
    return "other"

def main():
    csv_path = os.path.join(project_root, "notebooks", "sample_for_labeling_prelabeled.csv")
    df = pd.read_csv(csv_path)
    label_col = 'intent_label'
    df = df[df[label_col].notna()]
    
    df['predicted'] = df['text'].apply(classify_keyword)
    
    total = len(df)
    correct = (df['predicted'] == df[label_col]).sum()
    
    print(f"OVERALL ACCURACY: {correct}/{total} ({(correct/total)*100:.1f}%)\n")
    
    print(f"{'Intent':<25} | {'Count':<5} | {'Accuracy'}")
    print("-" * 50)
    for intent in sorted(df[label_col].unique()):
        sub = df[df[label_col] == intent]
        c = (sub['predicted'] == intent).sum()
        print(f"{intent:<25} | {len(sub):<5} | {(c/len(sub))*100:.1f}%")

if __name__ == "__main__":
    main()
