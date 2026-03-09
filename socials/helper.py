import requests
from django.conf import settings

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