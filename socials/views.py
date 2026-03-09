from django.shortcuts import render,redirect
from core.utils import encrypt_data, decrypt_data
from rest_framework.views import APIView
from django.http import JsonResponse, HttpResponseRedirect
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import SocialAccount
from accounts.helper import get_company
from accounts.models import Company
from .serializers import SocialAccountSerializer
import urllib.parse , requests
from uuid import UUID
from django.conf import settings
from django.urls import reverse
from .helper import subscribe_page_to_webhook


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

        state = encrypt_data(str(company.id))+','+encrypt_data(request.query_params.get('from',"web"))

        encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe='')
        encoded_scope = urllib.parse.quote(scope, safe='')
        encoded_state = urllib.parse.quote(state, safe='')

        fb_login_url = (
            f"https://www.facebook.com/v20.0/dialog/oauth"
            f"?client_id={meta_id}"
            f"&redirect_uri={encoded_redirect_uri}"
            f"&scope={encoded_scope}"
            f"&state={encoded_state}"
        )

        return Response({"redirect_url": fb_login_url})


class FacebookCallbackView(APIView):
    permission_classes = []
    
    def get(self, request):
        code = request.GET.get("code")
        error = request.GET.get("error")
        state_data = request.GET.get("state")

        # Decrypt state (company_id + source platform)
        try:
            state = UUID(decrypt_data(state_data.split(",")[0]))
            _from = decrypt_data(state_data.split(",")[1])
        except Exception as e:
            print(f"❌ State decryption failed: {e}")
            return HttpResponseRedirect(f"{settings.FRONTEND_URL}/user/integrations")

        print(f"Facebook callback: code={code}, error={error}, state={state_data}, from={_from}")

        if error:
            return HttpResponseRedirect(f"{settings.FRONTEND_URL}/user/integrations")

        if not code:
            return HttpResponseRedirect(f"{settings.FRONTEND_URL}/user/integrations")

        # ── Step 1: Exchange code for short-lived access token ──
        token_url = "https://graph.facebook.com/v20.0/oauth/access_token"
        redirect_uri = request.build_absolute_uri(reverse("facebook_callback")).split("?")[0]
        if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
            redirect_uri = redirect_uri.replace("http://", "https://")

        resp = requests.get(token_url, params={
            "client_id": settings.META_ID,
            "redirect_uri": redirect_uri,
            "client_secret": settings.META_SECRET,
            "code": code,
        })
        data = resp.json()

        if "access_token" not in data:
            return JsonResponse({"error": "Token exchange failed", "details": data})

        user_access_token = data["access_token"]

        # ── Debug: log granted scopes ──
        debug_resp = requests.get(
            "https://graph.facebook.com/v20.0/debug_token",
            params={
                "input_token": user_access_token,
                "access_token": f"{settings.META_ID}|{settings.META_SECRET}",
            }
        )

        # ── Step 2: Exchange for long-lived user token ──
        exchange_resp = requests.get(
            "https://graph.facebook.com/v20.0/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": settings.META_ID,
                "client_secret": settings.META_SECRET,
                "fb_exchange_token": user_access_token,
            }
        )
        long_lived_user_token = exchange_resp.json().get("access_token", user_access_token)

        # ── Step 3: Fetch all pages ──
        pages_resp = requests.get(
            "https://graph.facebook.com/v20.0/me/accounts",
            params={"access_token": long_lived_user_token}
        )
        pages_data = pages_resp.json()

        if "data" not in pages_data:
            return JsonResponse({"error": "Failed to fetch pages", "details": pages_data})

        pages = pages_data.get("data", [])
        if not pages:
            print("❌ No Facebook Pages found for this user token.")
            if _from == "app":
                return render(request, "redirect.html")
            return redirect(f"{settings.FRONTEND_URL}/user/integrations?error=no_pages")

        # ── Step 4: Resolve company ──
        try:
            company = Company.objects.get(id=state)
        except Company.DoesNotExist:
            return JsonResponse({"error": "Company not found"})

        # ── Step 5: Save each page & subscribe to webhooks ──

        for page in pages:
            page_id = page.get("id")
            page_name = page.get("name", "")
            page_access_token = page.get("access_token")

            if not page_id or not page_access_token:
                print(f"⚠️ Skipping page '{page_name}' — missing id or access_token.")
                continue

            SocialAccount.objects.update_or_create(
                account_id=page_id,
                platform="fb",
                defaults={
                    "company": company,
                    "name": page_name,
                    "token": page_access_token,
                }
            )
            print(f"✅ Facebook page '{page_name}' connected successfully.")

            subscribe_page_to_webhook(page_id, page_access_token,page_name)

            break

        if _from == "app":
            return render(request, "redirect.html")
        return redirect(f"{settings.FRONTEND_URL}/user/chat-profile")


class InstagramConnectView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):

        client_id = getattr(settings, 'IG_APP_ID')
        
        redirect_uri = request.build_absolute_uri(reverse("instagram_callback"))
        if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
            redirect_uri = redirect_uri.replace("http://", "https://")
        
        scope = "instagram_business_basic,instagram_business_manage_messages,instagram_business_manage_comments,instagram_business_content_publish,instagram_business_manage_insights"
        company = get_company(request.user)
        
        if not company:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        state = encrypt_data(str(company.id)) + "," + encrypt_data(request.query_params.get('from', "web"))

        encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe='')
        encoded_scope = urllib.parse.quote(scope, safe='')
        encoded_state = urllib.parse.quote(state, safe='')

        instagram_auth_url = (
            "https://www.instagram.com/oauth/authorize"
            f"?force_reauth=true"
            f"&client_id={client_id}"
            f"&redirect_uri={encoded_redirect_uri}"
            f"&scope={encoded_scope}"
            f"&response_type=code"
            f"&state={encoded_state}"
        )
        return Response({"redirect_url": instagram_auth_url})


class InstagramCallbackView(APIView):
    permission_classes = []

    def get(self, request):
        code = request.GET.get("code")
        error = request.GET.get("error")
        state_data = request.GET.get("state")

        if not state_data:
            return Response({"error": "State not found"}, status=status.HTTP_400_BAD_REQUEST)
        
        print("Instagram callback state data: ", state_data)
        
        state = UUID(decrypt_data(state_data.split(",")[0]))
        _from = decrypt_data(state_data.split(",")[1])

        if error or not code:
            return HttpResponseRedirect(f"{settings.FRONTEND_URL}/user/integrations")

        redirect_uri = request.build_absolute_uri(reverse("instagram_callback"))
        if not redirect_uri.endswith('/'):
            redirect_uri += '/'

        token_url = "https://api.instagram.com/oauth/access_token"
        app_id = str(getattr(settings, 'IG_APP_ID')).strip()
        app_secret = str(getattr(settings, 'IG_APP_SECRET')).strip()
        
        payload = {
            "client_id": app_id,
            "client_secret": app_secret,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code": code,
        }
        
        resp = requests.post(token_url, data=payload)
        data = resp.json()
        
        if "access_token" not in data:
            return JsonResponse({"error": "Instagram token exchange failed", "details": data}, status=400)

        user_access_token = data["access_token"]
        ig_user_id = data.get("user_id")
        
        exchange_url = "https://graph.instagram.com/access_token"
        exchange_params = {
            "grant_type": "ig_exchange_token",
            "client_secret": app_secret,
            "access_token": user_access_token,
        }
        
        exchange_resp = requests.get(exchange_url, params=exchange_params)
        exchange_data = exchange_resp.json()
        long_lived_token = exchange_data.get("access_token", user_access_token)

        # Try to resolve ID safely
        try:
            # 1. Fetch basic info (always available)
            ig_me_url = "https://graph.instagram.com/me"
            ig_me_params = {"fields": "id,username", "access_token": long_lived_token}
            ig_me_data = requests.get(ig_me_url, params=ig_me_params).json()
            
            final_ig_id = str(ig_me_data.get("id", ig_user_id))
            profile_name = ig_me_data.get("username", "Instagram Business")
            print(f"🔍 [Instagram Callback] Basic Data: {ig_me_data}")

            # 2. Optionally try to get ig_id (numeric business ID) without breaking if not available
            try:
                ig_id_params = {"fields": "ig_id", "access_token": long_lived_token}
                ig_id_resp = requests.get(ig_me_url, params=ig_id_params).json()
                if "ig_id" in ig_id_resp:
                    final_ig_id = str(ig_id_resp["ig_id"])
                    print(f"✨ [Instagram Callback] Found ig_id: {final_ig_id}")
                elif "error" in ig_id_resp:
                    print(f"ℹ️ [Instagram Callback] ig_id field not supported for this account: {ig_id_resp['error'].get('message')}")
            except Exception as e:
                print(f"ℹ️ [Instagram Callback] Skipping ig_id fetch: {e}")
                
        except Exception as e:
            print(f"❌ [Instagram Callback] Error during ID resolution: {e}")
            final_ig_id = str(ig_user_id)
            profile_name = "Instagram Business"

        try:
            company = Company.objects.get(id=state)
        except (Company.DoesNotExist, ValueError):
            return JsonResponse({"error": "Associated company not found"}, status=404)


        profile, created = SocialAccount.objects.update_or_create(
            platform='ig',
            company=company,
            defaults={
                "account_id": final_ig_id,
                "name": profile_name,
                "token": long_lived_token,
            }
        )
        print(f"✅ [Instagram] Profile {'created' if created else 'updated'}: {profile_name}")

        # 🔗 CRITICAL STEP: Subscribe the account to receive webhook messages
        # Instagram Login tokens ONLY work with graph.instagram.com (not graph.facebook.com)
        try:
            sub_url = "https://graph.instagram.com/v22.0/me/subscribed_apps"
            sub_res = requests.post(sub_url, params={
                "subscribed_fields": "messages,messaging_postbacks,messaging_seen",
                "access_token": long_lived_token
            }).json()
            
            if sub_res.get("success"):
                print(f"🔔 [Instagram] ✅ Webhook subscription SUCCESS for {profile_name}")
            else:
                print(f"⚠️ [Instagram] Subscription response: {sub_res}")
                # Fallback: Try via Facebook Graph API (for Page-linked accounts)
                try:
                    page_res = requests.get("https://graph.facebook.com/v22.0/me/accounts",
                                            params={"access_token": long_lived_token}).json()
                    for page in page_res.get("data", []):
                        page_sub = requests.post(
                            f"https://graph.facebook.com/v22.0/{page['id']}/subscribed_apps",
                            params={"subscribed_fields": "messages,messaging_postbacks,message_echoes",
                                    "access_token": page.get("access_token")}
                        ).json()
                        print(f"🔔 [Instagram] FB Page sub for {page['id']}: {page_sub}")
                except Exception as fb_e:
                    print(f"ℹ️ [Instagram] FB fallback subscription skipped: {fb_e}")

        except Exception as e:
            print(f"❌ [Instagram] Webhook subscription error: {e}")
        
        if _from == "app":
            return render(request, 'redirect.html')
        else:
            return redirect(f"{settings.FRONTEND_URL}/user/chat-profile")


