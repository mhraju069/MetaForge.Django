import requests
import os
from django.conf import settings
from .models import *

def subscribe_page_to_webhook(page_id, page_access_token,page_name):
    """Subscribe a Facebook Page to receive webhook events."""
    try:
        subscribe_url = f"https://graph.facebook.com/v20.0/{page_id}/subscribed_apps"
        params = {
            "subscribed_fields": "messages,messaging_postbacks,messaging_optins,message_echoes",
            "access_token": page_access_token,
        }
        response = requests.post(subscribe_url, params=params)
        result = response.json()
        if result.get("success"):
            print(f"✅ Facebook Page {page_name} subscribed to webhook successfully.")
            return True
        else:
            print(f"❌ Failed to subscribe Facebook Page {page_name}: {result}")
            return False
    except Exception as e:
        print(f"❌ Exception while subscribing Facebook Page {page_name}: {e}")
        return False


def chec_subscription(company):
    pass


def check_account(platform,account_id):
    return SocialAccount.objects.filter(platform=platform,account_id=account_id).first()


def train_post_embedding(post):
    """
    Generate semantic embedding for a single post using a Pro setup (Sync).
    Called automatically via signals when a post is saved.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️ Training skipped: OPENROUTER_API_KEY not found.")
        return

    try:
        # Use OpenAI-compatible embedding model endpoint
        # Replace with your preferred embedding provider URL if necessary
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "input": post.caption,
            "model": "text-embedding-3-small"
        }
        
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            vector = resp.json()["data"][0]["embedding"]
            # Update post without triggering the signal infinite loop
            SocialPost.objects.filter(id=post.id).update(vector=vector)
            print(f"✨ Successfully trained post {post.post_id}")
        else:
            print(f"⚠️ Embedding API error: {resp.text}")
            
    except Exception as e:
        print(f"❌ Exception in training post {post.post_id}: {e}")