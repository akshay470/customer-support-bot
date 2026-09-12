import os
import json
import time
import logging
from typing import Dict, List
import groq

logger = logging.getLogger(__name__)

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Try loading from .env if not found
    try:
        from dotenv import load_dotenv
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(project_root, ".env"))
        api_key = os.getenv("GROQ_API_KEY")
    except ImportError:
        pass

if not api_key:
    logger.warning("GROQ_API_KEY missing. LLM calls will fail.")

client = groq.Groq(api_key=api_key)

JUDGE_SYSTEM_PROMPT = """You are an expert customer service evaluator. 
Your task is to judge a drafted reply to a customer message.
You will be provided with:
1. The original CUSTOMER MESSAGE
2. The DRAFTED REPLY
3. The RETRIEVED CONTEXT (historical examples used to ground the reply)

Evaluate the drafted reply on the following 4 dimensions using a 1-5 scale (1=Poor, 5=Excellent):
- groundedness: Does the reply strictly use the provided retrieved context and avoid hallucinating false details (like fake URLs or policies)? (If no context was retrieved, rate how safely it handled the lack of info)
- relevance: Does the reply accurately address the customer's specific intent and core question/complaint?
- tone: Is the reply professional, polite, and brand-appropriate?
- completeness: Does the reply fully resolve the issue, or provide the exact appropriate next steps (e.g. asking for order number)?
- overall: Overall rating of the reply quality (1-5)

Return your evaluation EXACTLY as a JSON object matching this schema, completely unformatted (no markdown blocks, no markdown ticks, just raw JSON):
{
    "groundedness": 5,
    "relevance": 5,
    "tone": 5,
    "completeness": 5,
    "overall": 5,
    "judge_reasoning": "Quick explanation of why these scores were given, noting any specific flaws."
}
"""

def judge_reply(customer_message: str, drafted_reply: str, retrieved_context: List[str]) -> Dict:
    """
    Evaluates a generated reply using an LLM-as-judge pattern.
    """
    context_str = "\n".join([f"- {ctx}" for ctx in retrieved_context]) if retrieved_context else "None"
    
    user_prompt = f"""CUSTOMER MESSAGE:
{customer_message}

RETRIEVED CONTEXT:
{context_str}

DRAFTED REPLY:
{drafted_reply}
"""
    
    max_retries = 3
    for attempt in range(max_retries):
        # Enforce minimum 3s delay before every call
        time.sleep(3.0)
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b", # Reliable Groq model for reasoning/JSON
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Simple regex to extract JSON object
            import re
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(raw_content)
            
        except groq.APIStatusError as e:
            if e.status_code == 429: # Rate limit
                sleep_time = 3.0 + (2 ** attempt)
                e_str = str(e)
                import re
                match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', e_str)
                if match:
                    mins = int(match.group(1)) if match.group(1) else 0
                    secs = float(match.group(2))
                    sleep_time = mins * 60 + secs + 1.0 # Buffer 1s
                
                if sleep_time > 10.0:
                    logger.warning(f"Sleep time too large ({sleep_time}s), failing fast.")
                    return {
                        "groundedness": None, "relevance": None, "tone": None, "completeness": None, "overall": None,
                        "judge_reasoning": "Failed to judge due to API or Parsing Error."
                    }
                    
                logger.warning(f"Rate limited. Waiting {sleep_time}s before retry.")
                time.sleep(sleep_time)
            else:
                logger.error(f"Groq API Error: {e}")
                break
        except Exception as e:
            logger.error(f"Error during judging: {e}")
            break
            
    # Fallback response on failure
    return {
        "groundedness": None, "relevance": None, "tone": None, "completeness": None, "overall": None,
        "judge_reasoning": "Failed to judge due to API or Parsing Error."
    }
