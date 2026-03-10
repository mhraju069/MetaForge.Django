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
    if not api_key:
        print("⚠️ Company training skipped: OPENROUTER_API_KEY not found.")
        return

    # Prepare indexable content
    content = f"Company Name: {company.name}\n"
    content += f"Type: {company.type}\n"
    content += f"Description: {company.description}\n"
    content += f"Address: {company.address}\n"

    try:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "input": content,
            "model": "text-embedding-3-small"
        }
        
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            vector = resp.json()["data"][0]["embedding"]
            Company.objects.filter(id=company.id).update(vector=vector)
            print(f"✨ Successfully trained company {company.name}")
        else:
            print(f"⚠️ Company Embedding API error: {resp.text}")
            
    except Exception as e:
        print(f"❌ Exception in training company {company.name}: {e}")
