import requests
import os
from django.conf import settings
from .models import Company

def train_company_embedding(company):
    """
    Generate semantic embedding for company data (name, type, description, address).
    Called automatically via signals when a company is saved.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if (not api_key or not api_key.strip()) and (not openai_key or not openai_key.strip()):
        print("⚠️ Company training skipped: No AI Key found.")
        return

    # Prepare indexable content
    # Only include fields that have data
    parts = []
    if company.name: parts.append(f"Company Name: {company.name}")
    if company.type: parts.append(f"Type: {company.type}")
    if company.description: parts.append(f"Description: {company.description}")
    if company.address: parts.append(f"Address: {company.address}")
    
    content = "\n".join(parts)
    
    if not content.strip():
        return

    try:
        if openai_key and openai_key.strip():
            url = "https://api.openai.com/v1/embeddings"
            auth_key = openai_key.strip()
            model = "text-embedding-3-small"
        else:
            url = "https://openrouter.ai/api/v1/embeddings"
            auth_key = api_key.strip()
            model = "openai/text-embedding-3-small"

        headers = {"Authorization": f"Bearer {auth_key}"}
        payload = {
            "input": content,
            "model": model
        }
        
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            vector = resp.json()["data"][0]["embedding"]
            Company.objects.filter(id=company.id).update(vector=vector)
            print(f"✨ [Sync] Successfully trained company {company.name or 'Unnamed'}")
        else:
            print(f"⚠️ Company Embedding API error: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"❌ Exception in training company {company.name or 'Unnamed'}: {e}")
