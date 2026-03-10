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

async def generate_ai_reply(user_message: str):
    """Generate a reply using Rust-accelerated AI bridge."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "AI Key not found."
    
    if rust_ai:
        try:
            # Run the synchronous Rust function in a separate thread to keep FastAPI fast
            return await anyio.to_thread.run_sync(rust_ai.generate_reply, api_key, user_message)
        except Exception as e:
            print(f"❌ Rust AI Error: {e}")
            return f"Rust Error: {str(e)}"
    
    # Fallback to Python if Rust is not available
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant for MetaForge."},
                        {"role": "user", "content": user_message}
                    ],
                },
                timeout=30.0
            )
            res_data = response.json()
            return res_data['choices'][0]['message']['content']
        except Exception as e:
            return f"Python Fallback Error: {str(e)}"

@router.api_route("/{platform}/", methods=["GET", "POST"])
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
                    ai_reply = await generate_ai_reply(text)
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
