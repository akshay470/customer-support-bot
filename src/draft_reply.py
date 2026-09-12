import os
import sys
import json
import numpy as np
import time
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.classify import get_client

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Basic module caching
_MODEL = None
_EMBEDDINGS = None
_METADATA = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL

def load_index():
    global _EMBEDDINGS, _METADATA
    if _EMBEDDINGS is None or _METADATA is None:
        index_dir = os.path.join(project_root, "data", "resolved_threads_index")
        _EMBEDDINGS = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            _METADATA = json.load(f)
    return _EMBEDDINGS, _METADATA

def cosine_similarity(query_emb, doc_embs):
    # Dot product logic since norm affects similarity
    dot_products = np.dot(doc_embs, query_emb)
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    
    # Safe division
    norms = query_norm * doc_norms
    norms[norms == 0] = 1e-10
    
    return dot_products / norms

def draft_reply(customer_message: str, intent: str) -> dict:
    model = get_model()
    embeddings, metadata = load_index()
    
    q_emb = model.encode([customer_message])[0]
    sims = cosine_similarity(q_emb, embeddings)
    
    top_k = min(3, len(sims))
    if top_k == 0:
        return {"draft_reply": "No context available.", "retrieved_examples": [], "retrieval_similarity_scores": []}
        
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    retrieved_context = []
    thread_ids = []
    scores = []
    
    for i, idx in enumerate(top_indices):
        meta = metadata[idx]
        thread_ids.append(meta['thread_id'])
        scores.append(float(sims[idx]))
        
        context_str = f"Example {i+1}:\n"
        context_str += f"Customer Issue: {meta['customer_text']}\n"
        context_str += f"Agent Resolution: {meta['brand_resolution']}\n"
        retrieved_context.append(context_str)
        
    context_block = "\n".join(retrieved_context)
    
    system_prompt = f"""You are a helpful customer support agent for Amazon.
Your goal is to draft a reply to the customer's message based on how similar issues were resolved in the past.

The intent of the customer's request has been classified as: {intent}

Here are some past examples of how highly similar issues were successfully resolved:
{context_block}

Instructions:
1. Emulate the tone and style of the "Agent Resolution" from the examples, but tailor it strictly to the current customer's issue.
2. If the examples ask for verification (like a DM or details), you should do the same.
3. Be concise, polite, and directly helpful.
4. Output only the exact text of your drafted reply and nothing else. Do not output conversational filler or labels.
"""

    client = get_client()

    max_retries = 5
    for attempt in range(max_retries):
        # Enforce minimum 3s delay before every call
        time.sleep(3.0)
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": customer_message}
                ],
                timeout=15.0
            )
            draft = response.choices[0].message.content.strip()
            
            return {
                "draft_reply": draft,
                "retrieved_examples": thread_ids,
                "retrieval_similarity_scores": scores,
                "context_used": context_block
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 3.0 + (2 ** attempt)
                e_str = str(e)
                if "RateLimit" in type(e).__name__ or "rate_limit" in e_str or "429" in e_str:
                    import re
                    match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', e_str)
                    if match:
                        mins = int(match.group(1)) if match.group(1) else 0
                        secs = float(match.group(2))
                        sleep_time = mins * 60 + secs + 1.0 # Buffer 1s
                        
                if sleep_time > 10.0:
                    logger.warning(f"Sleep time too large ({sleep_time}s), failing fast.")
                    return {
                        "draft_reply": None,
                        "failed": True,
                        "error": str(e),
                        "retrieved_examples": thread_ids,
                        "retrieval_similarity_scores": scores,
                        "context_used": context_block
                    }
                        
                logger.warning(f"RAG Generation API error: {type(e).__name__}: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed after {max_retries} attempts. Final error: {type(e).__name__}: {e}")
                return {
                    "draft_reply": None,
                    "failed": True,
                    "error": str(e),
                    "retrieved_examples": thread_ids,
                    "retrieval_similarity_scores": scores,
                    "context_used": context_block
                }
