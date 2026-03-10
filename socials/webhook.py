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

async def generate_ai_reply(user_message: str, account_id: str = None):
    """Generate a reply using PRO-grade Vector Semantic Search."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "AI Key not found."

    # 1. Fetch Posts and Company Info (including vectors) from DB
    posts_data = []
    company_info = ""
    if account_id:
        try:
            posts_qs = await sync_to_async(lambda: list(SocialPost.objects.filter(account__account_id=account_id, vector__isnull=False).order_by("-created_at")[:200].values("post_id", "caption", "vector")))()
            posts_data = posts_qs
            
            # Fetch Company context
            account = await sync_to_async(SocialAccount.objects.get)(account_id=account_id)
            company = account.company
            if company.vector:
                company_info = f"\nCompany Profile: {company.name} - {company.description} at {company.type}.\n"
        except Exception as e:
            print(f"⚠️ Context fetch error: {e}")

    # 2. Get Vector for current User Message (Semantic Embedding)
    user_msg_vector = []
    if posts_data:
        async with httpx.AsyncClient() as client:
            try:
                emb_resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"input": user_message, "model": "text-embedding-3-small"}
                )
                if emb_resp.status_code == 200:
                    user_msg_vector = emb_resp.json()["data"][0]["embedding"]
            except Exception as e:
                print(f"⚠️ User message embedding failed: {e}")

    # 3. Utilize High-Performance Rust Semantic Search
    if rust_ai and user_msg_vector:
        try:
            # Struct our query with its vector and original text
            query_ctx = {"text": user_message, "vector": user_msg_vector}
            posts_json = json.dumps(posts_data)
            return await anyio.to_thread.run_sync(rust_ai.generate_reply, api_key, json.dumps(query_ctx), posts_json)
        except Exception as e:
            print(f"❌ Rust Vector Search Error: {e}")

    # 4. Fallback if Rust or Vectors are missing
    context = ""
    if posts_data:
        context = "\nRecent Posts Context:\n" + "\n".join([f"- {p['caption']}" for p in posts_data[:3]])

    system_prompt = "You are a professional shop assistant for MetaForge. Provide helpful and relevant replies. "
    if company_info:
        system_prompt += f"Background Shop Info: {company_info}"
    
    if context:
        system_prompt += f" Use this specific product/post context: \n{context}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                },
                timeout=30.0
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

@router.api_route("webhook/{platform}/", methods=["GET", "POST"])
async def unified_webhook_fastapi(platform: str, request: Request):
    if request.method == "GET":
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        if token == platform:
            return Response(content=challenge)
        return Response(content="Invalid verification token", status_code=403)

    if request.method == "POST":
        try:
            data = await request.json()
            print(f"🚀 [{platform}] FastAPI Webhook JSON: {json.dumps(data)}")

            if platform == "fb":
                entry = data.get("entry", [])
                if not entry: return {"status": "no_entry"}

                entry0 = entry[0]
                account_id = entry0.get("id")

                # Call Django ORM async
                account = await sync_check_account("fb", account_id)
                if not account:
                    print(f"❌ [Facebook] No active profile found for {account_id}")
                    return {"status": "no_profile"}
                
                print(f"✅ [Facebook] Account found: {account.name or account.account_id}")

                messaging = entry0.get("messaging", [])
                if not messaging: return {"status": "no_messaging"}

                msg_event = messaging[0]
                client_id = msg_event.get("sender", {}).get("id")
                text = msg_event.get("message", {}).get("text", "")
                
                if text:
                    print(f"📘 [Facebook] Message from {client_id}: {text}")
                    # GENERATE AI REPLY
                    print(f"🤖 Generating AI reply for: '{text}'...")
                    ai_reply = await generate_ai_reply(text, account_id)
                    print(f"✨ AI REPLY: {ai_reply}")

            return {"status": "received"}

        except Exception as e:
            print(f"❌ CRITICAL WEBHOOK ERROR ({platform}): {str(e)}")
            return {"status": "error", "message": str(e)}

# Keep Django version for compatibility if needed, but point it to FastAPI or just leave it
@csrf_exempt
def unified_webhook(request, platform):
    # This is the old Django view. 
    # You can redirect or keep it. For now, let's keep it as a fallback or remove it.
    return JsonResponse({"message": "Use the FastAPI endpoint at /api/socials/webhook/{platform}/"})
