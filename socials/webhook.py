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

# Bridge to Rust
try:
    import rust_ai
except ImportError:
    rust_ai = None

router = APIRouter()

# Helper to run Django ORM in async
sync_check_account = sync_to_async(check_account)

async def process_multimodal_description(media_url: str, media_type: str):
    """
    Use a Vision/Audio AI model to convert media into a text description 
    for semantic product search.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: return "No AI Key"

    # We use Gemini 2.0 Flash or GPT-4o-mini as they are excellent at multimodal
    model = "google/gemini-2.0-flash-001"
    
    prompt = "Describe this product in detail for an inventory search. Focus on: item type, color, material, and style. Be concise."
    if media_type == "audio":
        prompt = "Transcribe this audio message. If it describes a product, list its features (color, type, etc) for search."

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
    """Generate a reply using PRO-grade Vector Semantic Search and Multimodal support."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "AI Key not found."

    # 0. Multimodal Pre-processing: If user sent an image/audio, convert it to a searchable description
    actual_query = user_message
    if media_url:
        print(f"🖼️ [Agent] Processing {media_type} to find products...", flush=True)
        media_description = await process_multimodal_description(media_url, media_type)
        actual_query = f"{user_message} [User sent {media_type} described as: {media_description}]"
        print(f"📖 [Agent] Media Description: {media_description}", flush=True)

    # 1. Fetch Posts and Company Info (including vectors) from DB
    posts_data = []
    company_info = ""
    if account_id:
        try:
            # High-performance fetching of posts and company info in one sync block
            def fetch_context():
                posts = list(SocialPost.objects.filter(
                    account__account_id=account_id, 
                ).order_by("-created_at")[:20].values("post_id", "caption", "vector"))
                
                acc = SocialAccount.objects.select_related('company').get(account_id=account_id)
                comp = acc.company
                c_info = f"\nCompany Profile: {comp.name} - {comp.description} ({comp.type}).\n" if comp else ""
                
                return posts, c_info

            posts_data, company_info = await sync_to_async(fetch_context)()
        except Exception as e:
            print(f"⚠️ Context fetch error: {e}", flush=True)

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
        "1. ONLY use data from the 'Shop Context'. If a product or price is not there, say you cannot find it.\n"
        "2. Do NOT invent prices, currencies, or details (like ratings or fake websites).\n"
        "3. NEVER output JSON, code blocks, or special formatted lists. Talk as if you are in a chat.\n"
        "4. Use the specific currency (e.g., BDT) shown in the context.\n"
        "5. If you see a price like '1,630.00 BDT', use exactly that."
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
                client_id = msg_event.get("sender", {}).get("id")
                message = msg_event.get("message", {})
                
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
                    
                    # GENERATE AI REPLY (Now supports multimodal)
                    print(f"🤖 Generating AI reply for: '{incoming_log}'...", flush=True)
                    ai_reply = await generate_ai_reply(text, account_id, media_url, media_type)
                    print(f"✨ AI REPLY: {ai_reply}", flush=True)

            return {"status": "received"}

        except Exception as e:
            print(f"❌ CRITICAL WEBHOOK ERROR ({platform}): {str(e)}", flush=True)
            return {"status": "error", "message": str(e)}

# Keep Django version for debugging
@csrf_exempt
def unified_webhook(request, platform):
    print(f"⚠️ [Django Fallback] Webhook hit the Django view instead of FastAPI! Platform: {platform}", flush=True)
    return JsonResponse({"message": "This request was handled by Django. Please check ASGI config to route to FastAPI."})
