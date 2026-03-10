from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
import json, requests
from .models import *
from .helper import *
User = get_user_model()

@csrf_exempt
def unified_webhook(request, platform):

    if request.method == "GET":
        verify_token = platform
        token = request.GET.get("hub.verify_token")
        challenge = request.GET.get("hub.challenge")
        
        if token == verify_token:
            return HttpResponse(challenge)
        return HttpResponse("Invalid verification token", status=403)
    
    if request.method == "POST":
        raw_body = ""
        try:
            raw_body = request.body.decode("utf-8")
            
            data = json.loads(raw_body)
            print(f"🚀 [{platform}] Webhook JSON: {json.dumps(data)}")

            if platform == "fb":
                print(f"📘 [Facebook] Webhook received")
                entry = data.get("entry", [])
                if not entry:
                    return JsonResponse({"status": "no_entry"})

                entry0 = entry[0]
                account_id = entry0.get("id")

                account = check_account("fb",account_id)
                if not account:
                    print(f"❌ [Facebook] No active profile found for {account_id}")
                    return JsonResponse({"status": "no_profile"})
                
                print(f"✅ [Facebook] Account found: {account.name or account.account_id}")

                messaging = entry0.get("messaging", [])
                if not messaging:
                    return JsonResponse({"status": "no_messaging"})

                msg_event = messaging[0]
                client_id = msg_event.get("sender", {}).get("id")
                text = msg_event.get("message", {}).get("text", "")
                print(f"📘 [Facebook] Message from {client_id}: {text}")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL WEBHOOK ERROR ({platform}):")
            print(traceback.format_exc())
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

        return JsonResponse({"status": "received"})