from fastapi import APIRouter, Request, Response
from asgiref.sync import sync_to_async
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
import json, requests
from .models import *
from .helper import *

router = APIRouter()

# Helper to run Django ORM in async
sync_check_account = sync_to_async(check_account)

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
                print(f"📘 [Facebook] Message from {client_id}: {text}")

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
