import os
from fastapi import APIRouter, Request, Response
from asgiref.sync import sync_to_async
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
import json, requests, httpx
import anyio
from .models import *
from .helper import *
from .image_search import compute_phash_from_url, find_best_visual_match

# Bridge to Rust
try:
    import rust_ai
except ImportError:
    rust_ai = None

router = APIRouter()

# Global cache to prevent double processing of same mid (message id)
# In production, use Redis. For now, a memory set works.
PROCESSED_MIDS = set()

# Helper to run Django ORM in async
sync_check_account = sync_to_async(check_account)

async def send_facebook_message(recipient_id: str, message_text: str, images: list = None, access_token: str = ""):
    """Sends a text message and optional image attachments to a Facebook user."""
    if not access_token:
        print("❌ Cannot send message: No access token provided.")
        return

    fb_url = f"https://graph.facebook.com/v20.0/me/messages?access_token={access_token}"
    
    async with httpx.AsyncClient() as client:
        # 1. Send Text Reply
        if message_text:
            try:
                resp = await client.post(fb_url, json={
                    "recipient": {"id": recipient_id},
                    "message": {"text": message_text}
                })
                if resp.status_code != 200:
                    print(f"❌ FB Text Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                print(f"❌ Error sending FB text: {e}")

        # 2. Send Images
        if images:
            for img in images:
                try:
                    resp = await client.post(fb_url, json={
                        "recipient": {"id": recipient_id},
                        "message": {
                            "attachment": {
                                "type": "image",
                                "payload": {"url": img, "is_reusable": True}
                            }
                        }
                    })
                    if resp.status_code != 200:
                        print(f"❌ FB Image Error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    print(f"❌ Error sending FB image {img}: {e}")

async def process_multimodal_description(media_url: str, media_type: str):
    """
    Use a Vision/Audio AI model to convert media into a text description 
    for semantic product search.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: return "No AI Key"

    # We use Gemini 2.0 Flash or GPT-4o-mini as they are excellent at multimodal
    model = "google/gemini-2.0-flash-001"
    
    prompt = (
        "Analyze this content for a shop assistant task.\n"
        "1. If it's an IMAGE: Describe the product (type, color, style, patterns) in very high detail for internal inventory search. Reply with ONLY the description.\n"
        "2. If it's AUDIO: Transcribe the speech EXACTLY as spoken. If it's in Bengali, transcribe in Bengali.\n"
        "3. Determine the user's intent. If they show a product, describe it so I can find it in my database."
    )

    content_list = [{"type": "text", "text": prompt}]
    
    # For images, we send the URL. For audio, some models support URLs or we can use specific multimodal prompts.
    if media_type == "image":
        content_list.append({"type": "image_url", "image_url": {"url": media_url}})
    else:
        # Fallback for audio: Most multimodal models through OpenRouter prefer the URL in context if not direct
        content_list.append({"type": "text", "text": f"\n[Media Link to analyze: {media_url}]"})

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key.strip()}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content_list}]
                },
                timeout=30.0
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"❌ Multimodal Error: {resp.status_code} - {resp.text}")
                return "Could not analyze image."
        except Exception as e:
            print(f"❌ Multimodal Exception: {e}")
            return "Analysis failed."

async def generate_ai_reply(user_message: str, account_id: str = None, media_url: str = None, media_type: str = None):
    """Generate a reply using pHash Visual Search + Vector Semantic Search."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "AI Key not found."

    # STEP 1: Fetch Posts and Company Info from DB (with image hashes)
    posts_data = []
    company_info = ""
    social_account = None
    if account_id:
        try:
            def fetch_context():
                posts_qs = SocialPost.objects.filter(
                    account__account_id=account_id,
                    is_product=True,  # ✅ Only product posts for AI search
                ).order_by("-created_at")[:20].prefetch_related('media')

                results = []
                for p in posts_qs:
                    images_data = [
                        {"url": m.media_url, "hash": m.image_hash}
                        for m in p.media.all() if m.media_url
                    ]
                    results.append({
                        "post_id": p.post_id,
                        "caption": p.caption,
                        "vector": p.vector,
                        "images": images_data
                    })

                acc = SocialAccount.objects.select_related('company').get(account_id=account_id)
                comp = acc.company
                c_info = f"Company Profile: {comp.name} - {comp.description} ({comp.type})." if comp else ""
                return results, c_info, acc

            posts_data, company_info, social_account = await sync_to_async(fetch_context)()
        except Exception as e:
            print(f"⚠️ Context fetch error: {e}", flush=True)

    # STEP 2: USER SENT AN IMAGE → Try pHash visual match first (most accurate)
    if media_url and media_type == "image" and posts_data:
        print(f"🔎 [Visual] Attempting pHash visual search...", flush=True)
        query_hash = await anyio.to_thread.run_sync(compute_phash_from_url, media_url)
        if query_hash:
            print(f"🖼️ [Visual] Query image pHash: {query_hash}")
            visual_match = await anyio.to_thread.run_sync(
                find_best_visual_match, query_hash, posts_data, 15
            )
            if visual_match:
                caption = visual_match.get('caption', '')
                dist = visual_match.get('_visual_distance', 0)
                img_urls = [i['url'] for i in visual_match.get('images', []) if i.get('url')]
                print(f"✅ [Visual] EXACT visual match! Distance: {dist} | Caption: {caption[:60]}")

                # Build direct-reply prompt using verified matched product data
                matched_info = f"EXACT PRODUCT MATCH (Hamming distance: {dist}):\n{caption}"
                system_prompt = (
                    "You are a human Shop Assistant. The user sent a product image.\n"
                    "It was found in our inventory using visual fingerprint matching.\n"
                    "Reply NATURALLY confirming we have it. Include price and sizes from the product info.\n"
                    "At the END, add: IMAGE_URLS: followed by the URLs comma-separated.\n"
                    "DO NOT make up any info. Only use what is provided."
                )
                if company_info:
                    system_prompt += f"\n\nShop: {company_info}"
                system_prompt += f"\n\nProduct Info:\n{matched_info}"
                if img_urls:
                    system_prompt += f"\n\nIMAGE_URLS: {','.join(img_urls)}"

                async with httpx.AsyncClient() as client:
                    try:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key.strip()}"},
                            json={
                                "model": "google/gemini-2.0-flash-001",
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_message or "Found my product?"}
                                ]
                            },
                            timeout=25.0
                        )
                        if resp.status_code == 200:
                            return resp.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"❌ [Visual] LLM delivery error: {e}")

            else:
                print(f"⚠️ [Visual] No visual match within threshold. Falling back to AI description + text search.")
        else:
            print(f"⚠️ [Visual] pHash failed. Falling back to AI description.")

    # STEP 3: Audio OR no visual match → Use Multimodal description + text/vector search
    actual_query = user_message or ""
    if media_url and not (media_type == "image" and posts_data):
        print(f"🖼️ [Agent] Processing {media_type} via AI description...", flush=True)
        media_description = await process_multimodal_description(media_url, media_type)
        actual_query = f"{user_message} [User sent {media_type}: {media_description}]"
        print(f"📖 [Agent] Media Description: {media_description}", flush=True)
    elif media_url and media_type == "image":
        # Visual search failed - still describe image for text search
        print(f"🖼️ [Agent] Describing image for fallback text search...")
        media_description = await process_multimodal_description(media_url, media_type)
        actual_query = f"{user_message} [User sent image: {media_description}]"


    # 2. Get Vector for current User Message (Semantic Embedding)
    user_msg_vector = []
    if posts_data:
        openai_key = os.getenv("OPENAI_API_KEY")
        async with httpx.AsyncClient() as client:
            try:
                # Use OpenAI if key provided (direct), otherwise use OpenRouter's embedding model
                if openai_key and openai_key.strip():
                    url = "https://api.openai.com/v1/embeddings"
                    headers = {"Authorization": f"Bearer {openai_key.strip()}"}
                    model = "text-embedding-3-small"
                else:
                    url = "https://openrouter.ai/api/v1/embeddings"
                    headers = {"Authorization": f"Bearer {api_key.strip()}"}
                    model = "openai/text-embedding-3-small"

                emb_resp = await client.post(
                    url,
                    headers=headers,
                    json={"input": actual_query, "model": model}
                )
                if emb_resp.status_code == 200:
                    user_msg_vector = emb_resp.json()["data"][0]["embedding"]
                    print(f"🧬 Generated Vector for query: {actual_query[:40]}...")
                else:
                    print(f"⚠️ Embedding API error: {emb_resp.status_code} - {emb_resp.text}")
            except Exception as e:
                print(f"⚠️ Query embedding failed: {e}")

    # 3. Agentic Thought Process: Should we use Rust Search?
    final_context = ""
    if rust_ai and user_msg_vector:
        try:
            print("🤖 [Agent] Searching through shop posts via Rust Search Engine...")
            # Use our new Rust search tool
            vector_str = json.dumps(user_msg_vector)
            posts_json = json.dumps(posts_data)
            limit = 10
            
            # The tool call (Agentic Search)
            search_results = await anyio.to_thread.run_sync(rust_ai.search_context, vector_str, posts_json, limit)
            
            if search_results and search_results.strip():
                print(f"✅ [Agent] Relevant product context found by search engine.")
                final_context = search_results
            else:
                print(f"⚠️ [Agent] No semantically relevant products found for query. Using direct history.")
        except Exception as e:
            print(f"❌ [Agent] Search Tool Error: {e}")

    # 4. Fallback Context (Agentic Backup)
    if not final_context and posts_data:
        # If vector search finds nothing, we still look at latest posts as context
        final_context = "\nRecent Posts (Direct): " + "\n".join([f"- {p['caption']}" for p in posts_data[:10]])

    # 5. Core System Prompt (The Agent's instructions)
    system_prompt = (
        "You are an AI Shop Assistant for MetaForge. Talk naturally like a real person. "
        "Your only job is to provide product information from the 'Shop Context' below. "
        "\n\nSTRICT RULES:\n"
        "1. BE CRITICAL: Check the 'Match Confidence' score in the context. \n"
        "   - If Confidence > 0.70: You can say 'Yes, we have it!'\n"
        "   - If Confidence is between 0.45 and 0.70: Be cautious. Say 'I found something similar' and describe it instead of confirming a direct match.\n"
        "   - If Confidence < 0.45 or no results: Say 'I'm sorry, I couldn't find a match in our current inventory.'\n"
        "2. LANGUAGE MATCHING: Respond in the EXACT same language the user used.\n"
        "3. PRODUCT SEARCH: Carefully compare the user's request (or image description) with the 'Shop Context'.\n"
        "4. IMAGE_URLS: If you find a matching product, you MUST include ALL its image URLs. "
        "   At the VERY end of your response, add the tag 'IMAGE_URLS:' followed by ALL URLs separated by commas.\n"
        "5. DO NOT INVENT DATA: Never make up prices or availability.\n"
        "6. NO JSON: Talk like a human, no code blocks."
    )
    if company_info:
        system_prompt += f"\n\nShop Background: {company_info}"
    
    if final_context:
        system_prompt += f"\n\nShop Context:\n{final_context}"

    print(f"✉️ [Agent] Sending query to AI Model with {len(final_context)} chars of context...")

    # 6. Call LLM for final delivery
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": actual_query}
                ]
            }
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=25.0
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"]
                return answer
            else:
                print(f"❌ LLM error: {resp.status_code} - {resp.text}")
                return "I'm sorry, I'm having trouble retrieving details right now."
        except Exception as e:
            print(f"❌ AI delivery error: {e}")
            return "I'm having trouble connecting to my brain right now."

@router.api_route("/{platform}/", methods=["GET", "POST"])
async def unified_webhook_fastapi(platform: str, request: Request):
    print(f"📡 [FastAPI] Incoming request for platform: {platform}", flush=True)
    
    if request.method == "GET":
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        if token == platform:
            return Response(content=challenge)
        return Response(content="Invalid verification token", status_code=403)

    if request.method == "POST":
        try:
            data = await request.json()
            print(f"🚀 [{platform}] FastAPI Webhook JSON: {json.dumps(data)}", flush=True)

            if platform == "fb":
                entry = data.get("entry", [])
                if not entry: 
                    print("⚠️ No entry found in Facebook webhook", flush=True)
                    return {"status": "no_entry"}

                entry0 = entry[0]
                account_id = entry0.get("id")

                # Call Django ORM async
                account = await sync_check_account("fb", account_id)
                if not account:
                    print(f"❌ [Facebook] No active profile found for {account_id}", flush=True)
                    return {"status": "no_profile"}
                
                print(f"✅ [Facebook] Account found: {account.name or account.account_id}", flush=True)

                messaging = entry0.get("messaging", [])
                if not messaging: 
                    print("⚠️ No messaging events found", flush=True)
                    return {"status": "no_messaging"}

                msg_event = messaging[0]
                mid = msg_event.get("message", {}).get("mid")
                
                # 🛑 HALT: If we already processed this message ID, skip!
                if mid in PROCESSED_MIDS:
                    print(f"⏭️ [fb] Skipping already processed message: {mid}", flush=True)
                    return {"status": "already_processed"}
                
                # Add to cache (keep last 100 to save memory)
                PROCESSED_MIDS.add(mid)
                if len(PROCESSED_MIDS) > 100:
                    PROCESSED_MIDS.pop()

                sender_id = msg_event.get("sender", {}).get("id")
                recipient_id = msg_event.get("recipient", {}).get("id")
                message = msg_event.get("message", {})
                
                # 🛡️ IMPORTANT: Guard against Echo Messages and Meta Auto-Replies
                if message.get("is_echo") or sender_id == account_id:
                    print(f"🤫 [fb] Ignoring echo message from {sender_id}", flush=True)
                    return {"status": "ignored_echo"}

                client_id = sender_id
                text = message.get("text", "")
                attachments = message.get("attachments", [])
                
                media_url = None
                media_type = None
                
                if attachments:
                    att = attachments[0]
                    media_url = att.get("payload", {}).get("url")
                    media_type = att.get("type") # 'image', 'audio', 'video'
                    print(f"� [{platform}] Received {media_type}: {media_url}", flush=True)

                if text or media_url:
                    incoming_log = text if text else f"[{media_type} attachment]"
                    print(f"📘 [{platform}] Message from {client_id}: {incoming_log}", flush=True)
                    
                    # GENERATE AI REPLY (Now supports multimodal and image returns)
                    print(f"🤖 Generating AI reply for: '{incoming_log}'...", flush=True)
                    ai_reply = await generate_ai_reply(text, account_id, media_url, media_type)
                    
                    # Image sending logic
                    actual_text = ai_reply
                    extracted_images = []
                    if "IMAGE_URLS:" in ai_reply:
                        parts = ai_reply.split("IMAGE_URLS:")
                        actual_text = parts[0].strip()
                        extracted_images = [url.strip() for url in parts[1].split(",") if url.strip()]

                    if account: 
                        await send_facebook_message(client_id, actual_text, extracted_images, account.token)
                    
                    print(f"✨ AI REPLY SENT: {actual_text} (Images: {len(extracted_images)})", flush=True)

            return {"status": "received"}

        except Exception as e:
            print(f"❌ CRITICAL WEBHOOK ERROR ({platform}): {str(e)}", flush=True)
            return {"status": "error", "message": str(e)}

# Keep Django version for debugging
@csrf_exempt
def unified_webhook(request, platform):
    print(f"⚠️ [Django Fallback] Webhook hit the Django view instead of FastAPI! Platform: {platform}", flush=True)
    return JsonResponse({"message": "This request was handled by Django. Please check ASGI config to route to FastAPI."})
