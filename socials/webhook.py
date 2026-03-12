import os
import json
import datetime
import httpx
import anyio
from collections import OrderedDict
from fastapi import APIRouter, Request, Response
from asgiref.sync import sync_to_async
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .models import *
from .helper import *
from .image_search import compute_phash_from_url, find_best_visual_match

# ── Rust bridge ───────────────────────────────────────────────────────────────
try:
    import rust_ai
    print("✅ [Rust] rust_ai module loaded.", flush=True)
except ImportError:
    rust_ai = None
    print("⚠️ [Rust] rust_ai not available — falling back to Python context.", flush=True)

router = APIRouter()

# ── Dedup cache ───────────────────────────────────────────────────────────────
# OrderedDict preserves insertion order so we evict the OLDEST mid, not random.
PROCESSED_MIDS: OrderedDict = OrderedDict()
MAX_MIDS = 200

sync_check_account = sync_to_async(check_account)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _valid_key(k: str | None) -> str | None:
    """Return the key if non-empty, else None."""
    return k.strip() if k and k.strip() else None


async def _get_embedding(text: str, api_key: str, openai_key: str | None) -> list:
    """Fetch text embedding. Prefers OpenAI; falls back to OpenRouter."""
    if _valid_key(openai_key):
        url     = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {openai_key.strip()}"}
        model   = "text-embedding-3-small"
    elif _valid_key(api_key):
        url     = "https://openrouter.ai/api/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        model   = "openai/text-embedding-3-small"
    else:
        print("⚠️ [Embed] No API key available.")
        return []

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                url, headers=headers,
                json={"input": text, "model": model},
                timeout=15.0,
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            print(f"⚠️ [Embed] API error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️ [Embed] Exception: {e}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Facebook message sender
# ─────────────────────────────────────────────────────────────────────────────

async def send_facebook_message(
    recipient_id: str,
    message_text: str,
    images: list = None,
    access_token: str = "",
):
    if not access_token:
        print("❌ Cannot send message: No access token.")
        return

    fb_url = f"https://graph.facebook.com/v20.0/me/messages?access_token={access_token}"

    async with httpx.AsyncClient() as client:
        # Text message
        if message_text:
            try:
                resp = await client.post(fb_url, json={
                    "recipient": {"id": recipient_id},
                    "message":   {"text": message_text},
                }, timeout=10.0)
                if resp.status_code != 200:
                    print(f"❌ FB Text Error: {resp.status_code} - {resp.text}")
                else:
                    print(f"✅ [FB] Text sent to {recipient_id}")
            except Exception as e:
                print(f"❌ FB text send error: {e}")

        # Image attachments
        if images:
            for img_url in images:
                try:
                    resp = await client.post(fb_url, json={
                        "recipient": {"id": recipient_id},
                        "message": {
                            "attachment": {
                                "type": "image",
                                "payload": {"url": img_url, "is_reusable": True},
                            }
                        },
                    }, timeout=10.0)
                    if resp.status_code != 200:
                        print(f"❌ FB Image Error: {resp.status_code} - {resp.text}")
                    else:
                        print(f"✅ [FB] Image sent: {img_url[:60]}...")
                except Exception as e:
                    print(f"❌ FB image send error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal description (image / audio → text for search)
# ─────────────────────────────────────────────────────────────────────────────

async def process_multimodal_description(media_url: str, media_type: str) -> str:
    api_key = _valid_key(os.getenv("OPENROUTER_API_KEY"))
    if not api_key:
        return "No AI key configured."

    prompt = (
        "You are a product image analyzer for a shop assistant.\n"
        "Describe the product in this image in HIGH DETAIL: type, color, style, material, pattern.\n"
        "Focus on attributes useful for finding the product in a clothing/fashion inventory.\n"
        "Reply with ONLY the description, no other text."
    )
    content = [{"type": "text", "text": prompt}]
    if media_type == "image":
        content.append({"type": "image_url", "image_url": {"url": media_url}})
    else:
        content.append({"type": "text", "text": f"[Media URL: {media_url}]"})

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "google/gemini-2.0-flash-001",
                      "messages": [{"role": "user", "content": content}]},
                timeout=30.0,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            print(f"❌ [Multimodal] Error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"❌ [Multimodal] Exception: {e}")
    return "Could not analyze media."


# ─────────────────────────────────────────────────────────────────────────────
# Core AI reply generator
# ─────────────────────────────────────────────────────────────────────────────

async def generate_ai_reply(
    user_message: str,
    account_id: str = None,
    media_url: str = None,
    media_type: str = None,
    sender_id: str = None,
) -> str:
    """
    Full pipeline:
    1.  DB fetch  — product posts + company info
    2.  History   — 24-hour smart token window
    3.  Visual    — pHash search for images (temperature=0, no history)
    4.  Describe  — AI description of media for fallback text search
    5.  Embed     — vector embed the user query
    6.  Rust      — cosine similarity search (ultra-fast, in-process)
    7.  LLM       — generate reply with full context + history
    """
    api_key    = _valid_key(os.getenv("OPENROUTER_API_KEY"))
    openai_key = _valid_key(os.getenv("OPENAI_API_KEY"))

    if not api_key:
        return "AI is not configured. Please contact support."

    # ── 1. Fetch posts + company ──────────────────────────────────────────────
    posts_data    = []
    company_info  = ""
    company_name  = "our shop"      # safe fallback
    company_address = ""

    if account_id:
        try:
            def fetch_context():
                qs = SocialPost.objects.filter(
                    account__account_id=account_id,
                    is_product=True,                # only classified products
                ).order_by("-created_at")[:30].prefetch_related("media")

                results = []
                for p in qs:
                    imgs = [
                        {"url": m.media_url, "hash": m.image_hash or ""}
                        for m in p.media.all() if m.media_url
                    ]
                    results.append({
                        "post_id": p.post_id,
                        "caption": p.caption or "",
                        "vector":  p.vector,
                        "images":  imgs,
                    })

                acc  = SocialAccount.objects.select_related("company").get(account_id=account_id)
                comp = acc.company

                # FIX: Use FB page name (acc.name) as primary display name
                page_name = acc.name or ""
                if comp:
                    c_name = page_name or comp.name or "our shop"
                    c_type = comp.type        or ""
                    c_desc = comp.description or ""
                    c_addr = comp.address     or ""
                else:
                    c_name = page_name or "our shop"
                    c_type = c_desc = c_addr = ""

                parts = [p for p in [c_name, c_type, c_desc] if p]
                if c_addr:
                    parts.append(f"Address: {c_addr}")
                c_info = " | ".join(parts)
                return results, c_info, c_name, c_addr

            posts_data, company_info, company_name, company_address = \
                await sync_to_async(fetch_context)()
            print(f"🏪 [Company] '{company_name}' | {len(posts_data)} products", flush=True)
        except Exception as e:
            print(f"⚠️ [DB] Context fetch error: {e}", flush=True)

    # ── 2. Conversation history (24h, ~3000-char budget) ──────────────────────
    chat_history = []
    if account_id and sender_id:
        try:
            def fetch_history():
                acc  = SocialAccount.objects.get(account_id=account_id)
                conv, _ = Conversation.objects.get_or_create(account=acc, sender_id=sender_id)
                cutoff  = timezone.now() - datetime.timedelta(hours=24)
                msgs = list(reversed(
                    conv.messages.filter(created_at__gte=cutoff)
                                 .order_by("-created_at")[:20]
                ))
                # Trim to token budget (newest-first trim)
                budget, trimmed, total = 3000, [], 0
                for m in reversed(msgs):
                    cost = len(m.content)
                    if total + cost > budget:
                        break
                    trimmed.insert(0, m)
                    total += cost
                return trimmed

            msgs = await sync_to_async(fetch_history)()
            chat_history = [{"role": m.role, "content": m.content} for m in msgs]
            chars = sum(len(m["content"]) for m in chat_history)
            print(f"💬 [Memory] {len(chat_history)} msgs / {chars} chars")
        except Exception as e:
            print(f"⚠️ [Memory] Load failed: {e}")

    # ── 3. IMAGE → pHash visual search ───────────────────────────────────────
    # Design note: No history injected here.
    # When a pixel-level match exists, the product IS confirmed.
    # We use temperature=0 to eliminate all hallucination.
    # ─────────────────────────────────────────────────────────────────────────
    if media_url and media_type == "image" and posts_data:
        print("🔎 [Visual] pHash search...", flush=True)
        query_hash = await anyio.to_thread.run_sync(compute_phash_from_url, media_url)

        if query_hash:
            visual_match = await anyio.to_thread.run_sync(
                find_best_visual_match, query_hash, posts_data, 15
            )

            if visual_match:
                caption  = visual_match.get("caption", "")
                dist     = visual_match.get("_visual_distance", 0)
                img_urls = [i["url"] for i in visual_match.get("images", []) if i.get("url")]

                confidence = (
                    "IDENTICAL — pixel-perfect match" if dist == 0 else
                    "VERY HIGH confidence"            if dist <= 5 else
                    "HIGH confidence"
                )
                print(f"✅ [Visual] {confidence} | dist={dist} | {caption[:60]}")

                sys_prompt = (
                    f"You are the AI shopping assistant for '{company_name}'.\n\n"
                    "══ SITUATION ══\n"
                    "A customer sent a product image.\n"
                    f"Our visual fingerprint system found a {confidence} match "
                    f"(Hamming distance: {dist}/64).\n"
                    "The product HAS been identified. It IS in our inventory.\n\n"
                    "══ YOUR ONLY JOB ══\n"
                    "1. Confirm warmly that yes, we have this product.\n"
                    "2. Share the name, price, and sizes from the Product Data below.\n"
                    "3. Invite them to order or ask questions.\n\n"
                    "══ ABSOLUTE PROHIBITIONS ══\n"
                    "✗ NEVER ask for more pictures — product is already found.\n"
                    "✗ NEVER say you couldn't find it — you DID.\n"
                    "✗ NEVER ask what they're looking for — you already know.\n"
                    "✗ NEVER put image URLs inside text — use IMAGE_URLS tag only.\n"
                    "✗ NEVER invent prices, sizes, or data.\n\n"
                    "══ FORMAT ══\n"
                    "Natural, friendly confirmation message.\n"
                    "Last line must be exactly:\n"
                    "IMAGE_URLS: url1,url2\n"
                )
                if company_info:
                    sys_prompt += f"\n══ SHOP INFO ══\n{company_info}\n"
                sys_prompt += f"\n══ MATCHED PRODUCT DATA ══\n{caption}\n"
                if img_urls:
                    sys_prompt += f"\nProduct image URLs: {', '.join(img_urls)}\n"

                user_turn = "I sent a product image. Please confirm it and give me the details."
                if user_message:
                    user_turn += f" ({user_message})"

                async with httpx.AsyncClient() as client:
                    try:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json={
                                "model":    "google/gemini-2.0-flash-001",
                                "messages": [
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user",   "content": user_turn},
                                ],
                                "temperature": 0,
                            },
                            timeout=25.0,
                        )
                        if resp.status_code == 200:
                            return resp.json()["choices"][0]["message"]["content"]
                        print(f"❌ [Visual] LLM error {resp.status_code}: {resp.text[:200]}")
                    except Exception as e:
                        print(f"❌ [Visual] LLM exception: {e}")
                # Fall through to text search if LLM call failed
            else:
                print("⚠️ [Visual] No match within threshold. → text search")
        else:
            print("⚠️ [Visual] pHash failed. → text search")

    # ── 4. Media description for text/fallback search ─────────────────────────
    actual_query = user_message or ""
    if media_url:
        print(f"🖼️  AI description for {media_type}...", flush=True)
        desc = await process_multimodal_description(media_url, media_type)
        actual_query = f"{user_message or ''} [User sent {media_type}: {desc}]".strip()
        print(f"📖 Media description: {desc[:100]}", flush=True)

    # ── 5. Vector embedding ────────────────────────────────────────────────────
    user_vec = []
    if posts_data and actual_query:
        print("🧬 [Embed] Generating query vector...", flush=True)
        user_vec = await _get_embedding(actual_query, api_key, openai_key)
        if user_vec:
            print(f"🧬 [Embed] Vector ready (dim={len(user_vec)}) for: {actual_query[:60]}")

    # ── 6. Rust cosine search ──────────────────────────────────────────────────
    final_context = ""
    if rust_ai and user_vec:
        try:
            print("🤖 [Rust] Cosine search...", flush=True)
            result = await anyio.to_thread.run_sync(
                rust_ai.search_context,
                json.dumps(user_vec),
                json.dumps(posts_data),
                5,
            )
            if result and result.strip():
                print(f"✅ [Rust] Context: {len(result)} chars")
                final_context = result
            else:
                print("⚠️ [Rust] No products above 0.45 threshold.")
        except Exception as e:
            print(f"❌ [Rust] Search error: {e}")

    # Fallback: plain list with image URLs if Rust returned nothing
    if not final_context and posts_data:
        lines = []
        for p in posts_data[:10]:
            line = f"- {p['caption']}"
            imgs = [i["url"] for i in p.get("images", []) if i.get("url")]
            if imgs:
                line += f"\n  Image URLs: {', '.join(imgs)}"
            lines.append(line)
        final_context = "Available Products:\n" + "\n".join(lines)
        print(f"📦 [Fallback] Context built: {len(posts_data)} products", flush=True)

    # ── 7. LLM with history ────────────────────────────────────────────────────
    system_prompt = (
        f"You are the official AI shopping assistant for '{company_name}'.\n"
        f"You represent '{company_name}' exclusively.\n"
        "You have the full conversation history — use it to understand follow-up questions.\n\n"
        "══ CAPABILITIES ══\n"
        "✅ You CAN send product images to the customer.\n"
        "   Add at the VERY END of your reply (new line):\n"
        "   IMAGE_URLS: url1,url2\n"
        "   The system delivers images automatically — you do not need to explain this.\n\n"
        "══ RULES ══\n"
        f"1. IDENTITY: You are '{company_name}' assistant ONLY. Never mention competitors.\n"
        "2. HISTORY: Use conversation history to understand references like:\n"
        "   'that dress', 'the first one', 'its price', 'send picture', 'show me'.\n"
        "   When they ask for a picture of a discussed product, find its URLs in Shop Context\n"
        "   and include IMAGE_URLS at the end of your reply.\n"
        "3. IMAGES ON REQUEST: If user says 'send picture', 'show me', 'photo', 'ছবি পাঠাও',\n"
        "   find the product in Shop Context and ALWAYS include IMAGE_URLS.\n"
        "4. CONFIDENCE:\n"
        "   - Match > 0.70 → 'Yes, we have it!' + details + IMAGE_URLS\n"
        "   - Match 0.45–0.70 → 'I found something similar' + details + IMAGE_URLS\n"
        f"   - No match → apologize, say not found in {company_name}.\n"
        "5. LANGUAGE: Reply in the EXACT language the user used.\n"
        "6. IMAGE_URLS SOURCE: ONLY use URLs from 'Shop Context' section below.\n"
        "   ⛔ NEVER use URLs from conversation history (those are user-uploaded images).\n"
        "   ⛔ NEVER say 'I cannot send images' — you CAN, use the tag.\n"
        "7. NO INVENTED DATA: Never make up prices, sizes, or stock.\n"
        "8. NATURAL TONE: Speak like a friendly, knowledgeable shop assistant."
    )
    if company_info:
        system_prompt += f"\n\n── {company_name} Info ──\n{company_info}"
    if final_context:
        system_prompt += f"\n\n── Shop Context ──\n{final_context}"

    print(f"✉️  LLM call | context={len(final_context)}c | history={len(chat_history)}msgs")

    async with httpx.AsyncClient() as client:
        try:
            lm_msgs = [{"role": "system", "content": system_prompt}]
            lm_msgs.extend(chat_history)
            lm_msgs.append({"role": "user", "content": actual_query})

            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "google/gemini-2.0-flash-001", "messages": lm_msgs},
                timeout=30.0,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            print(f"❌ LLM error {resp.status_code}: {resp.text[:300]}")
            return "I'm having trouble right now. Please try again in a moment!"
        except Exception as e:
            print(f"❌ LLM exception: {e}")
            return "Connection issue. Please try again shortly!"


# ─────────────────────────────────────────────────────────────────────────────
# Webhook entry point
# ─────────────────────────────────────────────────────────────────────────────

@router.api_route("/{platform}/", methods=["GET", "POST"])
async def unified_webhook_fastapi(platform: str, request: Request):
    print(f"📡 [{platform}] {request.method}", flush=True)

    # ── GET: Facebook webhook verification ────────────────────────────────────
    if request.method == "GET":
        token     = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        if token == platform:
            return Response(content=challenge)
        return Response(content="Invalid verification token", status_code=403)

    # ── POST: Incoming messages ───────────────────────────────────────────────
    try:
        data = await request.json()
        print(f"🚀 [{platform}] Payload: {json.dumps(data)}", flush=True)

        if platform == "fb":
            entry = data.get("entry", [])
            if not entry:
                return {"status": "no_entry"}

            entry0     = entry[0]
            account_id = entry0.get("id")

            account = await sync_check_account("fb", account_id)
            if not account:
                print(f"❌ No FB account for id={account_id}", flush=True)
                return {"status": "no_profile"}
            print(f"✅ [FB] Account: {account.name or account.account_id}", flush=True)

            messaging = entry0.get("messaging", [])
            if not messaging:
                return {"status": "no_messaging"}

            msg_event = messaging[0]
            mid       = msg_event.get("message", {}).get("mid")

            # ── Deduplicate ──
            if mid in PROCESSED_MIDS:
                print(f"⏭️  Already processed: {mid}", flush=True)
                return {"status": "already_processed"}
            PROCESSED_MIDS[mid] = True
            if len(PROCESSED_MIDS) > MAX_MIDS:
                PROCESSED_MIDS.popitem(last=False)  # evict OLDEST

            sender_id = msg_event.get("sender", {}).get("id")
            message   = msg_event.get("message", {})

            # ── Skip echoes & self-sends ──
            if message.get("is_echo") or sender_id == account_id:
                print(f"🤫 Echo/self from {sender_id} — skipping", flush=True)
                return {"status": "ignored_echo"}

            client_id   = sender_id
            text        = message.get("text", "")
            attachments = message.get("attachments", [])
            media_url   = None
            media_type  = None

            if attachments:
                att        = attachments[0]
                media_url  = att.get("payload", {}).get("url")
                media_type = att.get("type")  # image / audio / video
                print(f"📎 {media_type}: {media_url}", flush=True)

            if not text and not media_url:
                return {"status": "no_content"}

            log_in = text if text else f"[{media_type}]"
            print(f"📘 From {client_id}: {log_in}", flush=True)

            # ── Generate AI reply ─────────────────────────────────────────────
            ai_reply = await generate_ai_reply(
                text, account_id, media_url, media_type, sender_id=client_id
            )

            # ── Parse IMAGE_URLS tag ──────────────────────────────────────────
            actual_text      = ai_reply
            extracted_images = []
            if "IMAGE_URLS:" in ai_reply:
                parts       = ai_reply.split("IMAGE_URLS:", 1)
                actual_text = parts[0].strip()
                extracted_images = [
                    u.strip() for u in parts[1].split(",") if u.strip()
                ]

            # ── Send to Facebook ──────────────────────────────────────────────
            await send_facebook_message(
                client_id, actual_text, extracted_images, account.token
            )

            # ── Save to conversation DB ───────────────────────────────────────
            try:
                def save_messages():
                    acc  = SocialAccount.objects.get(account_id=account_id)
                    conv, _ = Conversation.objects.get_or_create(
                        account=acc, sender_id=client_id
                    )
                    # Clean content — NEVER store raw CDN URLs (causes image loop bug)
                    if text:
                        user_content = text
                    elif media_type == "image":
                        user_content = "[User sent a product image]"
                    elif media_type == "audio":
                        user_content = "[User sent a voice message]"
                    else:
                        user_content = f"[User sent {media_type}]"

                    Message.objects.create(
                        conversation=conv, role="user",
                        content=user_content, media_url=None, media_type=media_type,
                    )
                    Message.objects.create(
                        conversation=conv, role="assistant", content=actual_text,
                    )
                    conv.save()

                await sync_to_async(save_messages)()
                print("💾 [Memory] Saved.", flush=True)
            except Exception as e:
                print(f"⚠️ [Memory] Save failed: {e}", flush=True)

            print(
                f"✨ Reply sent: {actual_text[:80]}... "
                f"(+{len(extracted_images)} images)",
                flush=True,
            )

        return {"status": "received"}

    except Exception as e:
        print(f"❌ WEBHOOK ERROR [{platform}]: {e}", flush=True)
        return {"status": "error", "message": str(e)}


# ── Django fallback (debug only) ──────────────────────────────────────────────
@csrf_exempt
def unified_webhook(request, platform):
    print(f"⚠️ [Django Fallback] platform={platform}", flush=True)
    return JsonResponse({"message": "Handled by Django fallback — check ASGI config."})
