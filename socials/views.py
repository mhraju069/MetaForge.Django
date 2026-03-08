from django.shortcuts import render
from core.utils import encrypt_token, decrypt_token
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import SocialAccount
from accounts.helper import get_company
from .serializers import SocialAccountSerializer
import urllib.parse
from uuid import UUID
from django.conf import settings
from django.urls import reverse
# Create your views here.


class SocialAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        company = get_company(request.user)
        social_accounts = SocialAccount.objects.filter(company=company)
        return Response(SocialAccountSerializer(social_accounts, many=True).data, status=status.HTTP_200_OK)
        

class FacebookConnectView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        meta_id = settings.META_ID
        redirect_uri = request.build_absolute_uri(reverse("facebook_callback"))
        if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
            redirect_uri = redirect_uri.replace("http://", "https://")
        
        scope = "pages_show_list,pages_manage_metadata,pages_read_engagement,pages_messaging"
        company = get_company(request.user)

        if not company:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)

        state = encrypt_token(str(company.id))
        _from = encrypt_token(request.query_params.get('from',"web"))

        fb_login_url = (
            f"https://www.facebook.com/v20.0/dialog/oauth"
            f"?client_id={meta_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={scope}"
            f"&state={state},{_from}"
        )

        return Response({"redirect_url": fb_login_url})


class FacebookCallbackView(APIView):
    permission_classes = []
    
    def get(self, request):
        code = request.GET.get("code")
        error = request.GET.get("error")
        state_data = request.GET.get("state")
        state = UUID(decrypt_token(state_data.split(",")[0]))
        _from = decrypt_token(state_data.split(",")[1])
        print(f"Facebook callback: code={code}, error={error}, state={state_data}, from={_from}")

        if error:
            return HttpResponseRedirect(f"{settings.FRONTEND_URL}/user/integrations")

        if not code:
            return HttpResponseRedirect(f"{settings.FRONTEND_URL}/user/integrations")

        token_url = "https://graph.facebook.com/v20.0/oauth/access_token"
        # Dynamic redirect_uri to match the one sent in ConnectView
        redirect_uri = request.build_absolute_uri(reverse("facebook_callback")).split('?')[0]
        if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
            redirect_uri = redirect_uri.replace("http://", "https://")
        
        params = {
            "client_id": settings.META_ID,
            "redirect_uri": redirect_uri,
            "client_secret": settings.META_SECRET,
            "code": code,
        }
        
        resp = requests.get(token_url, params=params)
        data = resp.json()
        
        if "access_token" not in data:
            return JsonResponse({"error": "Token exchange failed", "details": data})

        user_access_token = data["access_token"]


        # 1. Exchange short-lived User Access Token for Long-lived User Access Token
        exchange_url = "https://graph.facebook.com/v20.0/oauth/access_token"
        exchange_params = {
            "grant_type": "fb_exchange_token",
            "client_id": settings.META_ID,
            "client_secret": settings.META_SECRET,
            "fb_exchange_token": user_access_token,
        }
        
        exchange_resp = requests.get(exchange_url, params=exchange_params)
        exchange_data = exchange_resp.json()
        long_lived_user_token = exchange_data.get("access_token", user_access_token)

        # 2. Get pages with Long-lived Page Tokens using Long-lived User Token
        pages_url = "https://graph.facebook.com/v20.0/me/accounts"
        pages_resp = requests.get(pages_url, params={"access_token": long_lived_user_token})
        pages_data = pages_resp.json()

        if "data" not in pages_data:
            return JsonResponse({"error": "Failed to fetch pages", "details": pages_data})

        try:
            company = Company.objects.get(id=state)
        except Company.DoesNotExist:
            return JsonResponse({"error": "Company not found"})

        for page in pages_data["data"]:
            page_id = page["id"]
            page_name = page.get("name", "")
            page_access_token = page["access_token"]

            fb_profile, created = SocialAccount.objects.update_or_create(
                account_id=page_id,
                platform="fb",
                defaults={
                    "company": company,
                    "name": page_name,
                    "account_id": page_id,
                    "token": page_access_token,
                }
            )
            print("✅ Facebook page " + page_name + " connected successfully")
            break

        #subscribe page to webhook
        try:
            subscribe_url = f"https://graph.facebook.com/v20.0/{page_id}/subscribed_apps"
            params = {
                "subscribed_fields": "messages,messaging_postbacks,messaging_optins,message_echoes",
                "access_token": page_access_token
            }
            
            response = requests.post(subscribe_url, params=params)
            result = response.json()
            
            if result.get('success'):
                print(f"✅ Facebook Page subscribed successfully")
            else:
                print(f"❌ Error subscribing page: {result}")
                
        except Exception as e:
            print(f"❌ Error subscribing page: {e}")
        
        
        if _from == "app":
            return render(request,'redirect.html')
        else:
            return redirect(f"{settings.FRONTEND_URL}/user/chat-profile")