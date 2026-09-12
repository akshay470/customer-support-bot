import re
from typing import List, Dict

SENSITIVE_KEYWORDS = [
    "legal", "lawsuit", "refund", "fraud", "scam", "lawyer", "sue", 
    "stolen", "police", "compensation", "unauthorized", "chargeback"
]
# Regex to detect dollar amounts like $50, 50$, $50.00
DOLLAR_AMOUNT_REGEX = re.compile(r'(?:\$[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*\$|\b(?:dollars|bucks)\b)', re.IGNORECASE)

# Simple strong frustration language
FRUSTRATION_KEYWORDS = [
    "angry", "furious", "hate", "terrible", "worst", "pathetic", 
    "useless", "piece of shit", "wtf", "dammit", "ridiculous",
    "fucking", "bullshit"
]

def route_decision(
    customer_message: str, 
    intent: str, 
    confidence: str, 
    draft_reply: dict, 
    retrieval_similarity_scores: List[float]
) -> Dict[str, str]:
    """
    Decides whether to AUTO_HANDLE a message or ESCALATE it based on business rules.
    """
    customer_message_lower = customer_message.lower()
    
    # 1. Low intent confidence
    if confidence.lower() == "low":
        return {
            "decision": "ESCALATE",
            "reason": "Escalated: intent classification confidence is 'low'"
        }
        
    # 2. Inherently escalated intents
    if intent in ["agent_escalation", "other"]:
        return {
            "decision": "ESCALATE",
            "reason": f"Escalated: message belongs to inherently tricky intent category '{intent}'"
        }
        
    # 3. Retrieval similarity fallback
    # If no scores exist, or highest score is below 0.5
    if not retrieval_similarity_scores or max(retrieval_similarity_scores) < 0.5:
        max_score = max(retrieval_similarity_scores) if retrieval_similarity_scores else 0.0
        return {
            "decision": "ESCALATE",
            "reason": f"Escalated: highest retrieval similarity score {max_score:.2f} is below 0.5 threshold, no strong grounding example found"
        }
        
    # 4. Sensitive topic keywords
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in customer_message_lower:
            return {
                "decision": "ESCALATE",
                "reason": f"Escalated: sensitive topic keyword detected ('{keyword}')"
            }
            
    if DOLLAR_AMOUNT_REGEX.search(customer_message_lower):
        return {
            "decision": "ESCALATE",
            "reason": "Escalated: sensitive financial / dollar amount reference detected"
        }
        
    # 5. Strong negative sentiment/anger
    for kw in FRUSTRATION_KEYWORDS:
        if kw in customer_message_lower:
            return {
                "decision": "ESCALATE",
                "reason": f"Escalated: strong negative sentiment keyword detected ('{kw}')"
            }
            
    # Check for excessive exclamation marks or all-caps words
    if customer_message.count('!') >= 3:
        return {
            "decision": "ESCALATE",
            "reason": "Escalated: strong negative sentiment detected (multiple exclamation marks)"
        }
        
    # Simple all-caps check (words longer than 3 chars), explicitly ignoring "USER"
    words = [w for w in re.findall(r'\b[A-Z]{4,}\b', customer_message) if w != "USER"]
    if len(words) >= 2:
        return {
            "decision": "ESCALATE", 
            "reason": f"Escalated: strong negative sentiment detected (aggressive all-caps usage: {', '.join(words)})"
        }
        
    max_score = max(retrieval_similarity_scores) if retrieval_similarity_scores else 0.0
    
    # Otherwise
    return {
        "decision": "AUTO_HANDLE",
        "reason": f"Auto-Handled: intent confidence is {confidence}, grounding examples are strong (similarities up to {max_score:.2f}), and no sensitive/angry language detected."
    }
