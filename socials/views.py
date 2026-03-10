import os
import json
import urllib.parse
import httpx
import anyio
from uuid import UUID
from fastapi import APIRouter, Request, Response, Depends, HTTPException
from asgiref.sync import sync_to_async
from django.conf import settings
from .models import SocialAccount
from .serializers import SocialAccountSerializer
from .helper import subscribe_page_to_webhook, check_account
from accounts.helper import get_company
from accounts.models import Company
from core.utils import encrypt_data, decrypt_data

# Bridge to Rust
try:
    import rust_ai
except ImportError:
    rust_ai = None

router = APIRouter()

# --- Async Helpers ---
sync_get_company = sync_to_async(get_company)
sync_filter_social_accounts = sync_to_async(lambda company: list(SocialAccount.objects.filter(company=company)))
sync_get_company_by_id = sync_to_async(lambda id: Company.objects.get(id=id))
sync_update_or_create_social = sync_to_async(SocialAccount.objects.update_or_create)
sync_check_account = sync_to_async(check_account)

# --- FastAPI Routes ---

@router.get("/account/")
async def get_social_accounts(request: Request):
    # Note: In a real FastAPI app, we'd use a dependency for JWT auth.
    # For now, we assume the user is accessible via Django session if available
    # or we might need to implement JWT validation here.
    # Since this is a migration, I'll use a placeholder for auth or try to get it from request.
    user = getattr(request.state, "user", None) or await sync_to_async(lambda: request.scope.get("user"))()
    if not user or not user.is_authenticated:
        return {"error": "Not authenticated"}
    
    company = await sync_get_company(user)
    accounts = await sync_filter_social_accounts(company)
    return SocialAccountSerializer(accounts, many=True).data


@router.get("/connect/fb/")
async def facebook_connect(request: Request, _from: str = "web"):
    user = getattr(request.state, "user", None) or await sync_to_async(lambda: request.scope.get("user"))()
    if not user or not user.is_authenticated:
         return {"error": "Not authenticated"}

    company = await sync_get_company(user)
    if not company:
        return {"error": "Company not found"}

    meta_id = settings.META_ID
    # We use hardcoded callback or construct it
    redirect_uri = str(request.url_for("facebook_callback"))
    if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
        redirect_uri = redirect_uri.replace("http://", "https://")
    
    scope = "pages_show_list,pages_manage_metadata,pages_read_engagement,pages_messaging"
    state = encrypt_data(str(company.id)) + ',' + encrypt_data(_from)

    params = {
        "client_id": meta_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state
    }
    fb_login_url = f"https://www.facebook.com/v20.0/dialog/oauth?{urllib.parse.urlencode(params)}"

    return {"redirect_url": fb_login_url}


@router.get("/callback/fb/", name="facebook_callback")
async def facebook_callback(request: Request, code: str = None, error: str = None, state: str = None):
    if error or not code:
        return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/integrations"})

    try:
        company_id = UUID(decrypt_data(state.split(",")[0]))
        _from = decrypt_data(state.split(",")[1])
    except Exception:
        return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/integrations"})

    async with httpx.AsyncClient() as client:
        # Step 1: Token Exchange
        redirect_uri = str(request.url).split("?")[0]
        if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
            redirect_uri = redirect_uri.replace("http://", "https://")

        token_resp = await client.get("https://graph.facebook.com/v20.0/oauth/access_token", params={
            "client_id": settings.META_ID,
            "redirect_uri": redirect_uri,
            "client_secret": settings.META_SECRET,
            "code": code,
        })
        token_data = token_resp.json()
        if "access_token" not in token_data:
            return {"error": "Token exchange failed", "details": token_data}

        user_token = token_data["access_token"]

        # Step 2: Long-lived token
        exchange_resp = await client.get("https://graph.facebook.com/v20.0/oauth/access_token", params={
            "grant_type": "fb_exchange_token",
            "client_id": settings.META_ID,
            "client_secret": settings.META_SECRET,
            "fb_exchange_token": user_token,
        })
        long_lived_token = exchange_resp.json().get("access_token", user_token)

        # Step 3: Fetch Pages
        pages_resp = await client.get("https://graph.facebook.com/v20.0/me/accounts", params={"access_token": long_lived_token})
        pages_data = pages_resp.json()
        pages = pages_data.get("data", [])

        if not pages:
            if _from == "app": return {"status": "success", "message": "No pages found"} # Should render redirect.html
            return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/integrations?error=no_pages"})

        company = await sync_get_company_by_id(company_id)
        
        for page in pages:
            page_id = page.get("id")
            page_name = page.get("name", "")
            page_token = page.get("access_token")

            if page_id and page_token:
                await sync_update_or_create_social(
                    account_id=page_id,
                    platform="fb",
                    defaults={"company": company, "name": page_name, "token": page_token}
                )
                await sync_to_async(subscribe_page_to_webhook)(page_id, page_token, page_name)
                break # Only first page for now as per original code

    if _from == "app":
        return {"status": "success"} # Should be handled by frontend or template
    return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/chat-profile"})


@router.get("/connect/ig/")
async def instagram_connect(request: Request, _from: str = "web"):
    user = getattr(request.state, "user", None) or await sync_to_async(lambda: request.scope.get("user"))()
    if not user or not user.is_authenticated:
         return {"error": "Not authenticated"}

    company = await sync_get_company(user)
    if not company:
        return {"error": "Company not found"}

    client_id = settings.IG_APP_ID
    redirect_uri = str(request.url_for("instagram_callback"))
    if "ngrok" in redirect_uri and redirect_uri.startswith("http://"):
        redirect_uri = redirect_uri.replace("http://", "https://")

    scope = "instagram_business_basic,instagram_business_manage_messages,instagram_business_manage_comments,instagram_business_content_publish,instagram_business_manage_insights"
    state = encrypt_data(str(company.id)) + "," + encrypt_data(_from)

    params = {
        "force_reauth": "true",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "response_type": "code",
        "state": state
    }
    auth_url = f"https://www.instagram.com/oauth/authorize?{urllib.parse.urlencode(params)}"
    return {"redirect_url": auth_url}


@router.get("/callback/ig/", name="instagram_callback")
async def instagram_callback(request: Request, code: str = None, error: str = None, state: str = None):
    if error or not code:
        return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/integrations"})

    try:
        company_id = UUID(decrypt_data(state.split(",")[0]))
        _from = decrypt_data(state.split(",")[1])
    except Exception:
        return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/integrations"})

    async with httpx.AsyncClient() as client:
        redirect_uri = str(request.url).split("?")[0]
        if not redirect_uri.endswith('/'): redirect_uri += '/'
        
        # Step 1: Exchange code
        token_resp = await client.post("https://api.instagram.com/oauth/access_token", data={
            "client_id": settings.IG_APP_ID,
            "client_secret": settings.IG_APP_SECRET,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code": code,
        })
        token_data = token_resp.json()
        if "access_token" not in token_data:
             return {"error": "Instagram token exchange failed", "details": token_data}

        user_token = token_data["access_token"]
        
        # Step 2: Long-lived token
        exchange_resp = await client.get("https://graph.instagram.com/access_token", params={
            "grant_type": "ig_exchange_token",
            "client_secret": settings.IG_APP_SECRET,
            "access_token": user_token,
        })
        long_lived_token = exchange_resp.json().get("access_token", user_token)

        # Step 3: Me
        me_resp = await client.get("https://graph.instagram.com/me", params={"fields": "id,username", "access_token": long_lived_token})
        me_data = me_resp.json()
        final_ig_id = str(me_data.get("id", token_data.get("user_id")))
        username = me_data.get("username", "Instagram Business")

        company = await sync_get_company_by_id(company_id)
        await sync_update_or_create_social(
            platform='ig',
            company=company,
            defaults={"account_id": final_ig_id, "name": username, "token": long_lived_token}
        )

        # Step 4: Subscribe
        await client.post("https://graph.instagram.com/v22.0/me/subscribed_apps", params={
            "subscribed_fields": "messages,messaging_postbacks,messaging_seen",
            "access_token": long_lived_token
        })

    if _from == "app":
        return {"status": "success"}
    return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/chat-profile"})

