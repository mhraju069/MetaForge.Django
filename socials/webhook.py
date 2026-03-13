import os
import re
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
from .models import SocialAccount, SocialPost, PostMedia, Conversation, Message
from .helper import check_account
from .image_search import compute_phash_from_url, find_best_visual_match
from core.utils import decrypt_data

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



def _clean_llm_reply(text: str) -> str:
    """
    Strip AI artifacts that must never appear in a social media DM:
    - ```tool_code ... ``` blocks
    - ``` ... ``` fenced code blocks of any kind
    - Inline backtick code `like this`
    - Markdown bold (**text**) and italic (*text*)
    - Leading/trailing whitespace
    The IMAGE_URLS: tag at the end is intentionally preserved.
    """
    if not text:
        return text

    # Remove ```tool_code ... ``` blocks (may span multiple lines)
    text = re.sub(r"```tool_code[\s\S]*?```", "", text, flags=re.IGNORECASE)
    # Remove any other fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline backtick code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove markdown bold / italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove markdown headers (# ## ###)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



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
        print("❌ Cannot send message: No FB access token.")
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


async def send_instagram_message(
    recipient_id: str,
    message_text: str,
    images: list = None,
    access_token: str = "",
):
    """Send message to Instagram user via Facebook Graph API (Instagram Business)."""
    if not access_token:
        print("❌ Cannot send IG message: No access token.")
        return

    images = images or []

    # Instagram Business Messaging uses graph.facebook.com with a Page Access Token.
    # graph.instagram.com is for Instagram Login API — different flow.
    fb_url = f"https://graph.facebook.com/v20.0/me/messages?access_token={access_token}"
    ig_url = f"https://graph.instagram.com/v22.0/me/messages?access_token={access_token}"

    async with httpx.AsyncClient() as client:
        if message_text:
            payload = {
                "recipient": {"id": recipient_id},
                "message":   {"text": message_text},
            }
            sent = False

            # Try FB Graph first (works for Instagram Business / Connected Pages)
            try:
                resp = await client.post(fb_url, json=payload, timeout=15.0)
                if resp.status_code == 200:
                    print(f"✅ [IG→FB] Text sent to {recipient_id}", flush=True)
                    sent = True
                else:
                    print(f"⚠️  [IG→FB] {resp.status_code}: {resp.text[:300]}", flush=True)
            except Exception as e:
                print(f"⚠️  [IG→FB] Exception: {e}", flush=True)

            # If FB Graph failed, try IG Graph (Instagram Login API)
            if not sent:
                try:
                    resp2 = await client.post(ig_url, json=payload, timeout=15.0)
                    if resp2.status_code == 200:
                        print(f"✅ [IG→IG] Text sent to {recipient_id}", flush=True)
                        sent = True
                    else:
                        print(f"❌ [IG→IG] {resp2.status_code}: {resp2.text[:300]}", flush=True)
                except Exception as e:
                    print(f"❌ [IG→IG] Exception: {e}", flush=True)

            if not sent:
                print(f"❌ [IG] FAILED to send text to {recipient_id} — check token & permissions", flush=True)

        for img_url in images:
            img_payload = {
                "recipient": {"id": recipient_id},
                "message": {
                    "attachment": {
                        "type": "image",
                        "payload": {"url": img_url, "is_reusable": True},
                    }
                },
            }
            try:
                resp = await client.post(fb_url, json=img_payload, timeout=10.0)
                if resp.status_code != 200:
                    print(f"❌ [IG] Image error {resp.status_code}: {resp.text[:200]}", flush=True)
                else:
                    print(f"✅ [IG] Image sent: {img_url[:60]}...", flush=True)
            except Exception as e:
                print(f"❌ [IG] Image send exception: {e}", flush=True)



# ─────────────────────────────────────────────────────────────────────────────
# Multimodal description (image / audio → text for search)
# ─────────────────────────────────────────────────────────────────────────────

async def process_multimodal_description(media_url: str, media_type: str | None) -> str:
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
# Rust-powered post fetcher (non-blocking via anyio thread)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_posts_rust(api_url: str, access_token: str) -> dict:
    """
    Fetch posts from Facebook/Instagram Graph API using the Rust blocking client.
    Runs in a thread pool so it doesn't block the async event loop.
    Returns parsed JSON dict or empty dict on failure.
    """
    if not rust_ai:
        return {}
    try:
        raw = await anyio.to_thread.run_sync(
            rust_ai.fetch_meta_data, api_url, access_token
        )
        return json.loads(raw) if raw else {}
    except Exception as e:
        print(f"⚠️  [Rust Fetch] {e}", flush=True)
        return {}


# ─────────────────────────────────────────────────────────────────────────────

async def generate_ai_reply(
    user_message: str,
    account_id: str | None = None,
    media_url: str | None = None,
    media_type: str | None = None,
    sender_id: str | None = None,
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

                parts = [x for x in [c_name, c_type, c_desc] if x]
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

    # ── 3. IMAGE → pHash visual search ───────────────────────────────────────────
    # Threshold guide:
    #   0     = pixel-perfect identical image
    #   1–10  = same image, minor quality difference
    #   11–20 = same product, different photo (angle/lighting/background)
    #   21–30 = same style, different colour-way or slight design variation
    #   31+   = different product
    # We use 20 to be loose enough for angles but tight enough to separate products.
    PHASH_THRESHOLD = 20

    if media_url and media_type == "image" and posts_data:
        print("🔎 [Visual] pHash search...", flush=True)
        query_hash = await anyio.to_thread.run_sync(compute_phash_from_url, media_url)

        if query_hash:
            visual_match = await anyio.to_thread.run_sync(
                find_best_visual_match, query_hash, posts_data, PHASH_THRESHOLD
            )

            if visual_match:
                caption  = visual_match.get("caption", "")
                dist     = visual_match.get("_visual_distance", 0)
                img_urls = [i["url"] for i in visual_match.get("images", []) if i.get("url")]

                confidence = (
                    "IDENTICAL — pixel-perfect match" if dist == 0 else
                    "VERY HIGH confidence"             if dist <= 3  else
                    "HIGH confidence"                  if dist <= 12 else
                    "GOOD confidence (same product, likely different lighting/angle)"
                )
                print(f"✅ [Visual] {confidence} | dist={dist} | {caption[:60]}")

                sys_prompt = f"""You are the AI sales assistant for "{company_name}".
A customer just sent a product image and your visual matching system found a {confidence} match
(Hamming distance: {dist}/{PHASH_THRESHOLD} — lower means more similar).

The product HAS been identified. It IS in your inventory. Be excited about it!

YOUR RESPONSE MUST:
1. React naturally like a shop assistant who recognizes the product
   Example openers: "Oh yes, we have this!" / "Great choice!" / "Absolutely, this is one of ours!"
2. Share the product name, price, and available sizes/colors from Product Data
3. Include IMAGE_URLS on the VERY LAST LINE
4. Invite them to order: "Want to place an order? Just let me know! 😊"

ABSOLUTE RULES:
✗ NEVER say 'similar', 'something like this', 'closest match' — you have THE product
✗ NEVER ask for more pictures — product is already confirmed
✗ NEVER use markdown, bullet points, or code blocks
✗ NEVER invent prices, sizes, or availability
✗ NEVER put image URLs in text — only in IMAGE_URLS tag at the very end
✗ Keep it SHORT — 3-4 sentences + IMAGE_URLS

FORMAT:
Plain conversational text (2-4 sentences)
IMAGE_URLS: url1,url2
"""
                if company_info:
                    sys_prompt += f"\nSHOP: {company_info}\n"
                sys_prompt += f"\nPRODUCT DATA:\n{caption}\n"
                if img_urls:
                    sys_prompt += f"Product Image URLs: {', '.join(img_urls)}\n"
                if chat_history:
                    sys_prompt += "\nCONVERSATION CONTEXT (for reference):\n"
                    for m in chat_history[-4:]:  # last 4 turns for context
                        sys_prompt += f"{m['role'].upper()}: {m['content'][:100]}\n"

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
                            raw = resp.json()["choices"][0]["message"]["content"]
                            return _clean_llm_reply(raw)
                        print(f"❌ [Visual] LLM error {resp.status_code}: {resp.text[:200]}")
                    except Exception as e:
                        print(f"❌ [Visual] LLM exception: {e}")
                # Fall through to text search if LLM call failed
            else:
                print(f"⚠️  [Visual] No pHash match within threshold {PHASH_THRESHOLD}. → text+vision search")
        else:
            print("⚠️  [Visual] pHash failed. → text+vision search")

    # ── 4. Media description for text/fallback search ───────────────────────────────
    actual_query    = user_message or ""
    image_description = ""  # keep for LLM context injection later
    if media_url:
        print(f"🖼️  AI description for {media_type}...", flush=True)
        image_description = await process_multimodal_description(media_url, media_type)
        actual_query = f"{user_message or ''} [User sent {media_type}: {image_description}]".strip()
        print(f"📖 Media description: {image_description[:100]}", flush=True)

    # ── 5. Vector embedding ──────────────────────────────────────────────────────
    user_vec = []
    if posts_data and actual_query:
        print("🧬 [Embed] Generating query vector...", flush=True)
        user_vec = await _get_embedding(actual_query, api_key, openai_key)
        if user_vec:
            print(f"🧬 [Embed] Vector ready (dim={len(user_vec)}) for: {actual_query[:60]}")

    # ── 6. Rust cosine search ──────────────────────────────────────────────────────
    # Rust cosine search — threshold 0.35 (wider net, let LLM decide relevance)
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
                print("⚠️ [Rust] No products above threshold — using full inventory.")
        except Exception as e:
            print(f"❌ [Rust] Search error: {e}")

    # Fallback: always show ALL products when Rust found nothing
    # so AI always has inventory to talk about.
    if not final_context and posts_data:
        lines = []
        for p in posts_data[:15]:
            line = f"- {p['caption']}"
            imgs = [i["url"] for i in p.get("images", []) if i.get("url")]
            if imgs:
                line += f"\n  Image URLs: {', '.join(imgs)}"
            lines.append(line)
        final_context = "Full Inventory:\n" + "\n".join(lines)
        print(f"📦 [Fallback] Full inventory: {len(posts_data)} products", flush=True)

    # ── 7. LLM — World-class shop assistant prompt ──────────────────────────────
    # Philosophy: Think like a real shop owner who deeply cares about every customer.
    # The AI should feel like texting with a helpful, knowledgeable friend at the shop.
    # ──────────────────────────────────────────────────────────────────────────────

    # When user sent an image and pHash didn't find an exact match,
    # inject the image description as additional product finding context.
    image_context_note = ""
    if media_url and image_description:
        image_context_note = (
            f"\n\n══ CUSTOMER'S IMAGE ══\n"
            f"The customer sent a {media_type} image of a product they want.\n"
            f"AI vision analysis of their image: {image_description}\n\n"
            "Your task:\n"
            "  1. Compare the customer's image description to products in Shop Inventory below.\n"
            "  2. If a product matches features (colour, type, style), confirm we have it.\n"
            "  3. If the image is CLEARLY different (e.g. they sent a mini dress but inventory only has a maxi dress),\n"
            "     do NOT lie. Say you don't have that exact style but show them what you have that's closest.\n"
            "  4. If in doubt, stay helpful and positive."
        )

    has_products = bool(posts_data)

    system_prompt = f"""You are the personal AI sales assistant for "{company_name}".
You are the shop's most trusted team member. You know every product by heart.

══ YOUR PERSONALITY ══
• Warm, friendly, and professional Shop Owner mindset.
• Enthusiastic but HONEST — if we don't have the exact item, don't lie.
• Concise — 2-4 sentences max per reply.
• Multilingual — reply in the same language as the customer.

══ OUTPUT FORMAT (CRITICAL) ══
• Plain text ONLY — no markdown, no **bold**, no # headers, no bullet lists with dashes.
• NO code blocks, NO backticks, NO tool calls, NO function names.
• If sending product images, add on the VERY LAST LINE: IMAGE_URLS: url1,url2.

══ HOW TO HANDLE DIFFERENT SITUATIONS ══

[GREETINGS / hi / hello / assalamu alaikum]
→ Greet back warmly, introduce yourself briefly, ask how you can help.
→ Example: "Hey! Welcome to {company_name}! 😊 How can I help you today?"

[CUSTOMER ASKS WHAT YOU SELL / show products / what do you have]
→ Enthusiastically describe your product categories from inventory.
→ Pick 2-3 highlights and mention you can show photos + IMAGE_URLS.

[CUSTOMER SENDS AN IMAGE or asks about a specific product]
→ If it matches: "Oh yes! We carry this!" + Price/Details + IMAGE_URLS.
→ If no match: "We don't have that exact one right now, but check out this similar style!" + IMAGE_URLS.
→ Close with: "Want to order? Just let me know! 😊"

[CUSTOMER ASKS PRICE / SIZES / ORDERING]
→ Give exact data from inventory. Never invent data.
→ If ordering info is missing: "Just send your size and address here!"

══ SHOP OWNER TRICKS (FOR PERFECTION) ══
• UPSELL: "This is our top-seller right now!" or "People are loving this one!"
• CLOSING: "Should I save one for you?" or "Where should we deliver this?"
• EXCITEMENT: "Lovely Choice!", "Beautiful!", "Premium Quality!"

══ PRODUCT MATCHING RULES ══
• ACT LIKE A HUMAN: Don't claim a mini dress is a maxi dress.
• If Shop Inventory has items, always try to suggest the most relevant one.
• Confidently confirm "Yes we have this" ONLY if the description overlaps.
• NEVER fabricate or guess image URLs. Format: IMAGE_URLS: url1,url2"""

    if company_info:
        system_prompt += f"\n\n══ ABOUT {company_name.upper()} ══\n{company_info}"
    if final_context:
        system_prompt += f"\n\n══ SHOP INVENTORY ══\n{final_context}"
    if image_context_note:
        system_prompt += image_context_note
    if not has_products:
        system_prompt += (
            f"\n\n⚠️ NOTE: No products are currently in inventory for {company_name}. "
            "If asked about products, politely say you're setting up the catalog and "
            "invite them to check back soon or contact the shop directly."
        )

    print(f"✉️  LLM call | context={len(final_context)}c | history={len(chat_history)}msgs")

    async with httpx.AsyncClient() as client:
        try:
            lm_msgs = [{"role": "system", "content": system_prompt}]
            lm_msgs.extend(chat_history)
            lm_msgs.append({"role": "user", "content": actual_query})

            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": lm_msgs,
                    "temperature": 0.7,
                },
                timeout=30.0,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                return _clean_llm_reply(raw)
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

        if platform in ["fb", "ig"]:
            entry      = data.get("entry", [])
            entry0     = entry[0]
            account_id = entry0.get("id")

            messaging = entry0.get("messaging", [])
            if not messaging:
                return {"status": "no_messaging"}

            msg_event = messaging[0]
            message   = msg_event.get("message", {})
            mid       = message.get("mid") or msg_event.get("message_edit", {}).get("mid")
            sender_id = str(msg_event.get("sender", {}).get("id") or "")

            # ── 1. Deduplicate by MID ── সবার আগে, সবচেয়ে দ্রুত check
            if mid:
                if mid in PROCESSED_MIDS:
                    print(f"⏭️  Already processed MID: {mid}", flush=True)
                    return {"status": "already_processed"}
                PROCESSED_MIDS[mid] = True
                if len(PROCESSED_MIDS) > MAX_MIDS:
                    PROCESSED_MIDS.popitem(last=False)

            # ── 2. message_edit / read / delivery — reply দরকার নেই ──
            if ("message_edit" in msg_event and not message) or \
               "read" in msg_event or "delivery" in msg_event:
                return {"status": "event_ignored"}

            # ── 3. is_echo: bot নিজের sent message-এর echo — skip ──
            if message.get("is_echo"):
                print(f"🤫 is_echo from {sender_id} — skipping", flush=True)
                return {"status": "ignored_echo"}

            # ── 4. Account Resolve ──
            account = await sync_check_account(platform, account_id)

            if not account and platform == "ig":
                print(f"🔍 [Instagram] ID {account_id} not found in DB. Attempting auto-resolution...", flush=True)
                potential_accounts = await sync_to_async(
                    lambda: list(SocialAccount.objects.filter(platform="ig"))
                )()
                async with httpx.AsyncClient() as client:
                    for pot in potential_accounts:
                        try:
                            tok = decrypt_data(pot.token)
                            res = await client.get(
                                "https://graph.instagram.com/me",
                                params={"fields": "id,username", "access_token": tok},
                                timeout=10,
                            )
                            if res.status_code == 200:
                                me     = res.json()
                                f_id   = str(me.get("id"))
                                f_name = me.get("username", "")
                                if f_id == account_id or (f_name and f_name == pot.name):
                                    if pot.account_id != account_id:
                                        print(f"✨ [Instagram] Updating ID for {pot.name}: {pot.account_id} -> {account_id}", flush=True)
                                        pot.account_id = account_id
                                        await sync_to_async(pot.save)(update_fields=["account_id"])
                                    account = pot
                                    break
                        except Exception as e:
                            print(f"⚠️ Resolution error: {e}")

            if not account:
                print(f"❌ No {platform.upper()} account for {account_id}")
                return {"status": "no_profile"}

            # ── Refresh account from DB to get guaranteed-fresh token & account_id ──
            # auto-resolution may have just updated account_id in DB; re-fetch to sync.
            account = await sync_to_async(
                lambda: SocialAccount.objects.get(id=account.id)
            )()
            # Use the DB's current account_id (may differ from webhook's account_id after ID flip)
            resolved_account_id = account.account_id

            print(f"✅ [{platform.upper()}] Account: {account.name} | ID: {resolved_account_id}", flush=True)

            # ── 5. Company-র সব known bot ID সংগ্রহ করো (ID-flip এর বিরুদ্ধে) ──
            bot_ids: set = await sync_to_async(
                lambda: set(
                    str(x) for x in SocialAccount.objects.filter(
                        company=account.company, platform=platform
                    ).values_list("account_id", flat=True)
                )
            )()
            # webhook-এ আসা উভয় possible ID যোগ করো
            bot_ids.add(str(account_id))
            bot_ids.add(str(resolved_account_id))

            # ── 6. Sender যদি bot নিজেই হয়, skip ──
            if sender_id in bot_ids:
                print(f"🤫 Sender {sender_id} is a bot ID — skipping", flush=True)
                return {"status": "ignored_echo"}

            client_id   = sender_id
            text        = message.get("text", "")
            attachments = message.get("attachments", [])
            media_url   = None
            media_type  = None

            if not text and attachments:
                att        = attachments[0]
                media_url  = att.get("payload", {}).get("url")
                media_type = att.get("type")
                print(f"📎 {media_type}: {media_url}", flush=True)

            if not text and not media_url:
                return {"status": "no_content"}

            # ── 7. SELF-REPLY PROTECTION ──────────────────────────────────────
            # Instagram-এর ID-flip bug: bot-এর reply পাঠানোর পর Instagram
            # সেটা ভিন্ন account ID থেকে নতুন user message হিসেবে পাঠায়।
            # Company-র সব IG account-এর সর্বশেষ assistant message-এর
            # সাথে text মেলানো হয়। Match হলে skip করা হয়।
            if text:
                def _check_reflection():
                    last = (
                        Message.objects
                        .filter(
                            conversation__account__company=account.company,
                            conversation__account__platform=platform,
                            role="assistant",
                        )
                        .order_by("-created_at")
                        .first()
                    )
                    return bool(last and last.content.strip() == text.strip())

                if await sync_to_async(_check_reflection)():
                    print(f"🤫 Bot reply reflection ('{text[:30]}...') — skipping", flush=True)
                    return {"status": "ignored_reflection"}
            # ─────────────────────────────────────────────────────────────────

            log_in = text if text else f"[{media_type}]"
            print(f"📘 [{platform.upper()}] From {client_id}: {log_in}", flush=True)

            # ── 8. AI Reply Generate ──────────────────────────────────────────
            # resolved_account_id ব্যবহার করো — DB-তে যেটা আছে সেটাই সঠিক
            ai_reply = await generate_ai_reply(
                text or "", resolved_account_id, media_url, media_type, sender_id=client_id
            )

            # ── 9. IMAGE_URLS parse ───────────────────────────────────────────
            actual_text      = ai_reply
            extracted_images = []
            if "IMAGE_URLS:" in ai_reply:
                parts            = ai_reply.split("IMAGE_URLS:", 1)
                actual_text      = parts[0].strip()
                extracted_images = [u.strip() for u in parts[1].split(",") if u.strip()]

            # ── 10. Reply Send ────────────────────────────────────────────────
            # Fresh token and correct recipient_id essential here
            ig_token = decrypt_data(account.token)
            if platform == "fb" and client_id:
                await send_facebook_message(
                    client_id, actual_text, extracted_images, ig_token
                )
            elif platform == "ig" and client_id:
                await send_instagram_message(
                    client_id, actual_text, extracted_images, ig_token
                )

            # ── 11. DB-তে save ────────────────────────────────────────────────
            try:
                def save_messages():
                    acc  = SocialAccount.objects.get(id=account.id)
                    conv, _ = Conversation.objects.get_or_create(
                        account=acc, sender_id=client_id
                    )
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
                f"✨ Reply sent: {actual_text[:80]}... (+{len(extracted_images)} images)",
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
