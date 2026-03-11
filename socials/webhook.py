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
    if not api_key:
        return "No AI Key"

    model = "google/gemini-2.0-flash-001"

    prompt = (
        "Analyze this content for a shop assistant task.\n"
        "1. If it's an IMAGE: Describe the product (type, color, style, patterns) in very high detail for internal inventory search. Reply with ONLY the description.\n"
        "2. If it's AUDIO: Transcribe the speech EXACTLY as spoken. If it's in Bengali, transcribe in Bengali.\n"
        "3. Determine the user's intent. If they show a product, describe it so I can find it in my database."
    )

    content_list = [{"type": "text", "text": prompt}]

    if media_type == "image":
        content_list.append({"type": "image_url", "image_url": {"url": media_url}})
    else:
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


async def generate_ai_reply(
    user_message: str,
    account_id: str = None,
    media_url: str = None,
    media_type: str = None,
    sender_id: str = None,
):
    """Generate a reply using pHash Visual Search + Vector Semantic Search + Conversation Memory."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "AI Key not found."

    # ── STEP 1: Fetch product posts from DB ───────────────────────────────────
    posts_data = []
    company_info = ""
    social_account = None
    if account_id:
        try:
            def fetch_context():
                posts_qs = SocialPost.objects.filter(
                    account__account_id=account_id,
                    is_product=True,  # ✅ Only confirmed product posts
                ).order_by("-created_at")[:20].prefetch_related("media")

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
                        "images": images_data,
                    })

                acc = SocialAccount.objects.select_related("company").get(account_id=account_id)
                comp = acc.company
                c_info = f"Company: {comp.name} - {comp.description} ({comp.type})." if comp else ""
                return results, c_info, acc

            posts_data, company_info, social_account = await sync_to_async(fetch_context)()
        except Exception as e:
            print(f"⚠️ Context fetch error: {e}", flush=True)

    # ── STEP 2: Load Conversation History (Memory) ────────────────────────────
    chat_history = []
    if account_id and sender_id:
        try:
            def fetch_history():
                acc = SocialAccount.objects.get(account_id=account_id)
                conv, _ = Conversation.objects.get_or_create(account=acc, sender_id=sender_id)

                # Smart Window: last 20 msgs within 24 hours only
                # (older context is usually irrelevant for a shopping session)
                from django.utils import timezone
                import datetime
                cutoff = timezone.now() - datetime.timedelta(hours=24)

                msgs = conv.messages.filter(
                    created_at__gte=cutoff
                ).order_by("-created_at")[:20]  # Fetch more, then trim by token budget

                msgs = list(reversed(msgs))  # Oldest first for LLM

                # Token-aware trim: keep messages until we hit ~3000 chars (~750 tokens)
                # This prevents huge context windows blowing up API costs
                TOKEN_CHAR_BUDGET = 3000
                trimmed = []
                total_chars = 0
                for m in reversed(msgs):  # Start from most recent
                    msg_len = len(m.content)
                    if total_chars + msg_len > TOKEN_CHAR_BUDGET:
                        break
                    trimmed.insert(0, m)  # Prepend to maintain order
                    total_chars += msg_len

                return trimmed

            history_msgs = await sync_to_async(fetch_history)()
            for m in history_msgs:
                chat_history.append({"role": m.role, "content": m.content})
            if chat_history:
                print(f"💬 [Memory] Loaded {len(chat_history)} msgs ({sum(len(m['content']) for m in chat_history)} chars) from history")
            else:
                print("💬 [Memory] No recent history (fresh conversation or >24h gap)")
        except Exception as e:
            print(f"⚠️ [Memory] Failed to load history: {e}")

    # ── STEP 3: IMAGE → pHash Visual Search (pixel-level match) ──────────────
    if media_url and media_type == "image" and posts_data:
        print("🔎 [Visual] Attempting pHash visual search...", flush=True)
        query_hash = await anyio.to_thread.run_sync(compute_phash_from_url, media_url)
        if query_hash:
            print(f"🖼️ [Visual] Query pHash: {query_hash}")
            visual_match = await anyio.to_thread.run_sync(
                find_best_visual_match, query_hash, posts_data, 15
            )
            if visual_match:
                caption = visual_match.get("caption", "")
                dist = visual_match.get("_visual_distance", 0)
                img_urls = [i["url"] for i in visual_match.get("images", []) if i.get("url")]
                print(f"✅ [Visual] EXACT match! Distance: {dist} | {caption[:60]}")

                sys_prompt = (
                    "You are a human Shop Assistant. The user sent a product image.\n"
                    "It was EXACTLY matched in our inventory using visual fingerprint search.\n"
                    "Reply NATURALLY confirming we have it. Include price, sizes from the product info.\n"
                    "At the END of your reply, add: IMAGE_URLS: followed by URLs comma-separated.\n"
                    "DO NOT invent any data. Only use what is provided below."
                )
                if company_info:
                    sys_prompt += f"\n\nShop: {company_info}"
                sys_prompt += f"\n\nMatched Product:\n{caption}"
                if img_urls:
                    sys_prompt += f"\n\nIMAGE_URLS: {','.join(img_urls)}"

                async with httpx.AsyncClient() as client:
                    try:
                        lm_msgs = [{"role": "system", "content": sys_prompt}]
                        lm_msgs.extend(chat_history)
                        lm_msgs.append({"role": "user", "content": user_message or "Do you have this product?"})
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key.strip()}"},
                            json={"model": "google/gemini-2.0-flash-001", "messages": lm_msgs},
                            timeout=25.0,
                        )
                        if resp.status_code == 200:
                            return resp.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"❌ [Visual] LLM error: {e}")
            else:
                print("⚠️ [Visual] No match found. Falling back to text search.")
        else:
            print("⚠️ [Visual] pHash failed. Falling back to AI description.")

    # ── STEP 4: Text/Audio OR image fallback → description + vector search ────
    actual_query = user_message or ""
    if media_url:
        print(f"🖼️ [Agent] Getting AI description for {media_type}...", flush=True)
        media_description = await process_multimodal_description(media_url, media_type)
        actual_query = f"{user_message} [User sent {media_type}: {media_description}]"
        print(f"📖 [Agent] Description: {media_description}", flush=True)

    # ── STEP 5: Vector Embedding of user query ─────────────────────────────────
    user_msg_vector = []
    if posts_data and actual_query:
        openai_key = os.getenv("OPENAI_API_KEY")
        async with httpx.AsyncClient() as client:
            try:
                if openai_key and openai_key.strip():
                    emb_url = "https://api.openai.com/v1/embeddings"
                    emb_headers = {"Authorization": f"Bearer {openai_key.strip()}"}
                    emb_model = "text-embedding-3-small"
                else:
                    emb_url = "https://openrouter.ai/api/v1/embeddings"
                    emb_headers = {"Authorization": f"Bearer {api_key.strip()}"}
                    emb_model = "openai/text-embedding-3-small"

                emb_resp = await client.post(
                    emb_url, headers=emb_headers,
                    json={"input": actual_query, "model": emb_model}
                )
                if emb_resp.status_code == 200:
                    user_msg_vector = emb_resp.json()["data"][0]["embedding"]
                    print(f"🧬 Generated Vector for: {actual_query[:40]}...")
                else:
                    print(f"⚠️ Embedding error: {emb_resp.status_code}")
            except Exception as e:
                print(f"⚠️ Embedding failed: {e}")

    # ── STEP 6: Rust Vector Search or text fallback ────────────────────────────
    final_context = ""
    if rust_ai and user_msg_vector:
        try:
            print("🤖 [Agent] Rust vector search...")
            search_results = await anyio.to_thread.run_sync(
                rust_ai.search_context,
                json.dumps(user_msg_vector),
                json.dumps(posts_data),
                10
            )
            if search_results and search_results.strip():
                print("✅ [Agent] Context found via Rust search.")
                final_context = search_results
            else:
                print("⚠️ [Agent] No relevant products found.")
        except Exception as e:
            print(f"❌ [Agent] Rust search error: {e}")

    if not final_context and posts_data:
        final_context = "Recent Products:\n" + "\n".join([f"- {p['caption']}" for p in posts_data[:10]])

    # ── STEP 7: Build system prompt + inject history + call LLM ───────────────
    system_prompt = (
        "You are an AI Shop Assistant. Talk naturally like a real person. "
        "Use the 'Shop Context' to answer product questions. "
        "You have full conversation history - use it to understand follow-up questions.\n\n"
        "STRICT RULES:\n"
        "1. USE HISTORY: If user refers to something earlier ('that dress', 'the first one', 'its price'), use history to understand context.\n"
        "2. BE CRITICAL on confidence scores:\n"
        "   - Score > 0.70: Confirm 'Yes, we have it!'\n"
        "   - Score 0.45-0.70: Say 'I found something similar'\n"
        "   - Score < 0.45 or nothing: Say you could not find it.\n"
        "3. LANGUAGE: Reply in the EXACT language the user used.\n"
        "4. IMAGE_URLS: If product found, append 'IMAGE_URLS: url1,url2' at very end of reply.\n"
        "5. NO INVENTED DATA: Never make up prices or sizes.\n"
        "6. NO JSON: Speak like a human."
    )
    if company_info:
        system_prompt += f"\n\nShop Info: {company_info}"
    if final_context:
        system_prompt += f"\n\nShop Context:\n{final_context}"

    print(f"✉️ [Agent] Context: {len(final_context)} chars | History: {len(chat_history)} msgs")

    async with httpx.AsyncClient() as client:
        try:
            lm_msgs = [{"role": "system", "content": system_prompt}]
            lm_msgs.extend(chat_history)  # 📚 Full conversation history injected here
            lm_msgs.append({"role": "user", "content": actual_query})

            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key.strip()}"},
                json={"model": "google/gemini-2.0-flash-001", "messages": lm_msgs},
                timeout=25.0,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
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

                # 🛑 Deduplicate: skip already-processed message IDs
                if mid in PROCESSED_MIDS:
                    print(f"⏭️ [fb] Skipping already processed: {mid}", flush=True)
                    return {"status": "already_processed"}

                PROCESSED_MIDS.add(mid)
                if len(PROCESSED_MIDS) > 100:
                    PROCESSED_MIDS.pop()

                sender_id = msg_event.get("sender", {}).get("id")
                recipient_id = msg_event.get("recipient", {}).get("id")
                message = msg_event.get("message", {})

                # 🛡️ Guard against Echo Messages and Meta Auto-Replies
                if message.get("is_echo") or sender_id == account_id:
                    print(f"🤫 [fb] Ignoring echo from {sender_id}", flush=True)
                    return {"status": "ignored_echo"}

                client_id = sender_id
                text = message.get("text", "")
                attachments = message.get("attachments", [])

                media_url = None
                media_type = None

                if attachments:
                    att = attachments[0]
                    media_url = att.get("payload", {}).get("url")
                    media_type = att.get("type")  # 'image', 'audio', 'video'
                    print(f"📎 [{platform}] Received {media_type}: {media_url}", flush=True)

                if text or media_url:
                    incoming_log = text if text else f"[{media_type} attachment]"
                    print(f"📘 [{platform}] Message from {client_id}: {incoming_log}", flush=True)

                    # 🤖 GENERATE AI REPLY (with conversation memory + visual search)
                    print(f"🤖 Generating AI reply for: '{incoming_log}'...", flush=True)
                    ai_reply = await generate_ai_reply(
                        text, account_id, media_url, media_type, sender_id=client_id
                    )

                    # Parse IMAGE_URLS tag from reply
                    actual_text = ai_reply
                    extracted_images = []
                    if "IMAGE_URLS:" in ai_reply:
                        parts = ai_reply.split("IMAGE_URLS:")
                        actual_text = parts[0].strip()
                        extracted_images = [u.strip() for u in parts[1].split(",") if u.strip()]

                    # 📤 Send reply to Facebook
                    if account:
                        await send_facebook_message(client_id, actual_text, extracted_images, account.token)

                    # 💾 Save to conversation history DB
                    try:
                        def save_messages():
                            acc = SocialAccount.objects.get(account_id=account_id)
                            conv, _ = Conversation.objects.get_or_create(account=acc, sender_id=client_id)
                            # User message
                            user_content = text if text else f"[{media_type}: {media_url}]"
                            Message.objects.create(
                                conversation=conv,
                                role="user",
                                content=user_content,
                                media_url=media_url,
                                media_type=media_type,
                            )
                            # AI reply
                            Message.objects.create(
                                conversation=conv,
                                role="assistant",
                                content=actual_text,
                            )
                            conv.save()  # Update updated_at

                        await sync_to_async(save_messages)()
                        print(f"💾 [Memory] Saved messages to conversation.", flush=True)
                    except Exception as e:
                        print(f"⚠️ [Memory] Save failed: {e}", flush=True)

                    print(f"✨ AI REPLY SENT: {actual_text} (Images: {len(extracted_images)})", flush=True)

            return {"status": "received"}

        except Exception as e:
            print(f"❌ CRITICAL WEBHOOK ERROR ({platform}): {str(e)}", flush=True)
            return {"status": "error", "message": str(e)}


# Keep Django version for debugging
@csrf_exempt
def unified_webhook(request, platform):
    print(f"⚠️ [Django Fallback] Webhook hit Django view! Platform: {platform}", flush=True)
    return JsonResponse({"message": "This request was handled by Django. Please check ASGI config."})
