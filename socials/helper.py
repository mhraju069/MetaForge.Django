import requests
import os
from django.conf import settings
from .models import *


def detect_is_product(post) -> bool:
    """
    Use AI to detect if a social media post is about a product (for sale).
    Returns True if it's a product post, False otherwise.
    Uses a cheap/fast binary classification prompt.
    """
    if not post.caption or not post.caption.strip():
        return False

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    caption = post.caption.strip()[:500]  # Limit to 500 chars to save cost

    prompt = (
        "You are a product classifier for a social media shop manager.\n"
        "Read this social media post caption and answer ONLY with 'YES' or 'NO'.\n"
        "Question: Is this post about a product that is being sold or offered for sale?\n"
        "Rules:\n"
        "- YES: if it mentions a product with price, size, material, or availability\n"
        "- NO: if it is a greeting, announcement, story, celebration, or non-product content\n"
        f"\nCaption:\n{caption}\n\nAnswer (YES or NO only):"
    )

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            is_product = answer.startswith("YES")
            print(f"🤖 [ProductDetect] Post {post.post_id}: '{caption[:40]}...' → {'✅ PRODUCT' if is_product else '❌ NOT PRODUCT'}")
            return is_product
        else:
            print(f"⚠️ [ProductDetect] API error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ [ProductDetect] Exception: {e}")
        return False


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
    if post.vector:
        return 

    api_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if (not api_key or not api_key.strip()) and (not openai_key or not openai_key.strip()):
        return

    try:
        url = "https://openrouter.ai/api/v1/embeddings"
        auth_key = api_key.strip() if api_key else ""
        model = "openai/text-embedding-3-small"

        if openai_key and openai_key.strip():
            url = "https://api.openai.com/v1/embeddings"
            auth_key = openai_key.strip()
            model = "text-embedding-3-small"

        headers = {"Authorization": f"Bearer {auth_key}"}
        payload = {"input": post.caption, "model": model}
        
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            vector = resp.json()["data"][0]["embedding"]
            SocialPost.objects.filter(id=post.id).update(vector=vector)
            print(f"✨ [Sync] Successfully trained post {post.post_id}")
        else:
            print(f"⚠️ Sync Embedding API error: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"❌ Exception in training post {post.post_id}: {e}")


def train_post_image_hash(post):
    """
    Generate pHash fingerprints for all images in a post and save them to PostMedia.
    This enables pixel-level visual similarity search.
    Called via signals when a post or PostMedia is saved.
    """
    from .image_search import compute_phash_from_url

    media_items = PostMedia.objects.filter(post=post, image_hash__isnull=True)
    if not media_items.exists():
        return

    print(f"🖼️ [pHash] Generating image fingerprints for post {post.post_id}...")
    updated = 0
    for m in media_items:
        if m.media_url:
            h = compute_phash_from_url(m.media_url)
            if h:
                PostMedia.objects.filter(id=m.id).update(image_hash=h)
                updated += 1
                print(f"  ✅ [pHash] Saved hash for media {m.id}: {h}")
    
    print(f"  📊 [pHash] {updated} image fingerprints saved for post {post.post_id}")
