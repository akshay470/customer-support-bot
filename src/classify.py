"""
classify.py
Core module for intention classification using Groq API and structured JSON output.
"""
import os
import time
import json
import logging
from dotenv import load_dotenv
from groq import Groq

from src.intents import INTENTS
try:
    from src.few_shots import FEW_SHOTS
except ImportError:
    FEW_SHOTS = {}

logger = logging.getLogger(__name__)

# Build the system prompt dynamically from INTENTS
taxonomy_str = "\n".join([f"- {k}: {v}" for k, v in INTENTS.items()])

few_shots_str = ""
if FEW_SHOTS:
    few_shots_str = "\nHere are some examples to guide you:\n"
    for intent, examples in FEW_SHOTS.items():
        for ex in examples:
            few_shots_str += f"Example of {intent}: '{ex}' -> {intent}\n"

SYSTEM_PROMPT = f"""You are a customer support intent classifier.
Categorize the user's message into exactly ONE of the following categories:
{taxonomy_str}
{few_shots_str}
You must respond in strict JSON format with exactly two keys:
1. "intent": The chosen category name.
2. "confidence": "high", "medium", or "low" based on how directly the reasoning matched your chosen category.

Example output:
{{"intent": "delivery_delay", "confidence": "high"}}
"""

def get_client():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("No GROQ_API_KEY found in .env")
        return None
    return Groq(api_key=api_key)

def classify_intent(text: str, client=None, max_retries=3) -> dict:
    if client is None:
        client = get_client()
        if client is None:
            return {"intent": "other", "confidence": "low"}
            
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                timeout=15.0
            )
            val = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting block
            if val.startswith("```json"):
                val = val[7:]
            if val.endswith("```"):
                val = val[:-3]
            val = val.strip()
            
            # attempt to parse json
            result = json.loads(val)
            
            # ensure keys exist
            intent = result.get("intent", "other")
            confidence = result.get("confidence", "low")
            
            # Validate intent against taxonomy just in case
            if intent not in INTENTS:
                intent = "other"
                
            return {"intent": intent, "confidence": confidence}
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e}. Output was: {val}")
            return {"intent": "other", "confidence": "low"}
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logger.warning(f"API error: {type(e).__name__}: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed after {max_retries} attempts. Final error: {type(e).__name__}: {e}")
                return {"intent": "other", "confidence": "low"}
