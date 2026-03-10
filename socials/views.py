import os
import json
import urllib.parse
import httpx
import anyio
from uuid import UUID
from fastapi import APIRouter, Request, Response, Depends, HTTPException
from asgiref.sync import sync_to_async
from django.conf import settings
from .models import SocialAccount, SocialPost, PostMedia
from .serializers import SocialAccountSerializer
from .helper import subscribe_page_to_webhook, check_account
from accounts.helper import get_company
from accounts.models import Company
from core.utils import encrypt_data, decrypt_data
from rest_framework_simplejwt.authentication import JWTAuthentication

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

# --- Authentication Helper ---
async def get_authenticated_user(request: Request):
    """
    Unified user authentication for FastAPI.
    Checks request.state, request.scope, and manual JWT header.
    """
    # 1. Check if already attached (middleware)
    user = getattr(request.state, "user", None)
    if user and user.is_authenticated:
        return user
    
    # 2. Check ASGI scope (Django)
    scope_user = request.scope.get("user")
    if scope_user:
        if callable(scope_user):
            user = await sync_to_async(scope_user)()
        else:
            user = scope_user
        if user and user.is_authenticated:
            return user

    # 3. Manual JWT check (Compatible with SimpleJWT)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            jwt_auth = JWTAuthentication()
            validated_token = await sync_to_async(jwt_auth.get_validated_token)(token)
            user = await sync_to_async(jwt_auth.get_user)(validated_token)
            if user and user.is_active:
                return user
        except Exception as e:
            # print(f"Auth error: {e}")
            pass

    return None

async def generate_vector(text: str):
    """Generate vector for a given text using OpenAI or OpenRouter."""
    if not text or not text.strip():
        return None
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if (not api_key or not api_key.strip()) and (not openai_key or not openai_key.strip()):
        return None

    async with httpx.AsyncClient() as client:
        try:
            if openai_key and openai_key.strip():
                url = "https://api.openai.com/v1/embeddings"
                headers = {"Authorization": f"Bearer {openai_key.strip()}"}
                model = "text-embedding-3-small"
            else:
                url = "https://openrouter.ai/api/v1/embeddings"
                headers = {"Authorization": f"Bearer {api_key.strip()}"}
                model = "openai/text-embedding-3-small"

            resp = await client.post(
                url, 
                headers=headers, 
                json={"input": text, "model": model},
                timeout=10.0
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            else:
                print(f"⚠️ Embedding fetch failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"⚠️ Vector generation error: {e}")
    return None

# --- FastAPI Routes ---

@router.get("/account/")
async def get_social_accounts(request: Request):
    user = await get_authenticated_user(request)
    if not user:
        return {"error": "Not authenticated"}
    
    company = await sync_get_company(user)
    accounts = await sync_filter_social_accounts(company)
    return SocialAccountSerializer(accounts, many=True).data


@router.get("/fetch-posts/{account_id}/")
async def fetch_posts(account_id: str, request: Request):
    user = await get_authenticated_user(request)
    if not user:
        return {"error": "Not authenticated"}

    company = await sync_get_company(user)
    try:
        # Get account for this company
        account = await sync_to_async(SocialAccount.objects.get)(account_id=account_id, company=company)
    except Exception as e:
        return {"error": f"Account not found: {str(e)}"}

    if account.platform == "fb":
        return await sync_facebook_all_posts(account)
    elif account.platform == "ig":
        return await sync_instagram_all_posts(account)
    
    return {"error": "Platform not supported for fetching posts"}


    return {"error": "Platform not supported for fetching posts"}


async def sync_facebook_all_posts(account):
    """Sync all posts from a Facebook Page."""
    async with httpx.AsyncClient() as client:
        # fields: id, message (caption), full_picture, attachments (image/sub-images)
        fb_url = f"https://graph.facebook.com/v20.0/{account.account_id}/posts"
        params = {
            "fields": "id,message,created_time,full_picture,attachments{media_type,media,subattachments}",
            "access_token": account.token,
            "limit": 50
        }
        
        resp = await client.get(fb_url, params=params)
        data = resp.json()
        print(f"🚀 [Facebook API Response Data]: {json.dumps(data, indent=2)}")
        
        if "data" not in data:
            return {"error": "FB API Error", "details": data}
        
        fetched_count = 0
        for item in data["data"]:
            post_id = item.get("id")
            caption = item.get("message", "") # FB calls it message
            print(f"📝 Saving FB Post: {post_id} | Caption: {caption[:50]}...")

            # --- GENERATE VECTOR ---
            vector = await generate_vector(caption)
            if vector:
                print(f"🧬 Generated Vector for FB Post: {post_id}")

            # 1. Update/Create post
            post, created = await sync_to_async(SocialPost.objects.update_or_create)(
                account=account,
                post_id=post_id,
                defaults={"caption": caption, "vector": vector}
            )
            print(f"✅ Post {'Created' if created else 'Updated'}: {post_id}")

            # 2. Media Handling
            media_urls = set()
            
            # Check attachments for images/carousels (Prioritize attachments)
            attachments = item.get("attachments", {}).get("data", [])
            if attachments:
                for att in attachments:
                    # If it's a carousel (album), it has subattachments
                    sub = att.get("subattachments", {}).get("data", [])
                    if sub:
                        print(f"📦 Found Carousel (Album) with {len(sub)} items")
                        for s in sub:
                            src = s.get("media", {}).get("image", {}).get("src")
                            if src: media_urls.add(src)
                    else:
                        # Single image/video attachment
                        src = att.get("media", {}).get("image", {}).get("src")
                        if src: media_urls.add(src)
            
            # If no attachments found, fallback to full_picture
            if not media_urls and item.get("full_picture"):
                media_urls.add(item["full_picture"])

            # 3. Save media links
            media_count = 0
            for m_url in media_urls:
                _, m_created = await sync_to_async(PostMedia.objects.get_or_create)(
                    post=post,
                    media_url=m_url
                )
                if m_created: media_count += 1
            
            if media_count > 0:
                print(f"🖼️ Saved {media_count} new media items for post {post_id}")

            fetched_count += 1

        return {"status": "success", "count": fetched_count}


async def sync_instagram_all_posts(account):
    """Sync all media from Instagram Business Account."""
    async with httpx.AsyncClient() as client:
        # https://developers.facebook.com/docs/instagram-api/reference/ig-user/media/
        url = f"https://graph.facebook.com/v20.0/{account.account_id}/media"
        params = {
            "fields": "id,caption,media_type,media_url,thumbnail_url,children{media_url,media_type}",
            "access_token": account.token,
            "limit": 50
        }
        
        resp = await client.get(url, params=params)
        data = resp.json()
        print(f"📸 [Instagram API Response Data]: {json.dumps(data, indent=2)}")
        
        if "data" not in data:
            return {"error": "IG API Error", "details": data}
        
        fetched_count = 0
        for item in data["data"]:
            post_id = item.get("id")
            caption = item.get("caption", "")
            print(f"📝 Saving IG Post: {post_id} | Caption: {caption[:50]}...")

            # --- GENERATE VECTOR ---
            vector = await generate_vector(caption)
            if vector:
                print(f"🧬 Generated Vector for IG Post: {post_id}")

            post, created = await sync_to_async(SocialPost.objects.update_or_create)(
                account=account,
                post_id=post_id,
                defaults={"caption": caption, "vector": vector}
            )
            print(f"✅ Post {'Created' if created else 'Updated'}: {post_id}")

            # Media Handling
            media_urls = set()
            
            # For carousels (Prioritize children)
            children = item.get("children", {}).get("data", [])
            if children:
                print(f"📦 Found Instagram Carousel with {len(children)} items")
                for child in children:
                    if child.get("media_url"): media_urls.add(child["media_url"])
            elif item.get("media_url"):
                # Single photo/video
                media_urls.add(item["media_url"])
                
            media_count = 0
            for m_url in media_urls:
                _, m_created = await sync_to_async(PostMedia.objects.get_or_create)(
                    post=post,
                    media_url=m_url
                )
                if m_created: media_count += 1
            
            if media_count > 0:
                print(f"🖼️ Saved {media_count} new media items for post {post_id}")

            fetched_count += 1

        return {"status": "success", "count": fetched_count}


@router.get("/connect/fb/")
async def facebook_connect(request: Request, _from: str = "web"):
    user = await get_authenticated_user(request)
    if not user:
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
                account, _ = await sync_update_or_create_social(
                    account_id=page_id,
                    platform="fb",
                    defaults={"company": company, "name": page_name, "token": page_token}
                )
                await sync_to_async(subscribe_page_to_webhook)(page_id, page_token, page_name)
                
                # --- AUTO-FETCH POSTS AFTER SUCCESSFUL CONNECTION ---
                try:
                    await sync_facebook_all_posts(account)
                    print(f"🎯 Auto-fetched posts for FB Page: {page_name}")
                except Exception as e:
                    print(f"⚠️ Error auto-fetching FB posts: {e}")
                
                break # Only first page for now as per original code

    if _from == "app":
        return {"status": "success"} # Should be handled by frontend or template
    return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/chat-profile"})


@router.get("/connect/ig/")
async def instagram_connect(request: Request, _from: str = "web"):
    user = await get_authenticated_user(request)
    if not user:
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
        account, _ = await sync_update_or_create_social(
            platform='ig',
            company=company,
            defaults={"account_id": final_ig_id, "name": username, "token": long_lived_token}
        )

        # Step 4: Subscribe
        await client.post("https://graph.instagram.com/v22.0/me/subscribed_apps", params={
            "subscribed_fields": "messages,messaging_postbacks,messaging_seen",
            "access_token": long_lived_token
        })

        # --- AUTO-FETCH POSTS AFTER SUCCESSFUL CONNECTION ---
        try:
            await sync_instagram_all_posts(account)
            print(f"🎯 Auto-fetched posts for Instagram: {username}")
        except Exception as e:
            print(f"⚠️ Error auto-fetching IG posts: {e}")

    if _from == "app":
        return {"status": "success"}
    return Response(status_code=302, headers={"Location": f"{settings.FRONTEND_URL}/user/chat-profile"})


# --- Django Legacy Views (For URL Namespace compatibility) ---
from rest_framework.views import APIView
from rest_framework.response import Response as DRFResponse
from rest_framework import status as drf_status

class SocialAccountView(APIView):
    def get(self, request): return DRFResponse({"message": "FastAPI handled"})

class FacebookConnectView(APIView):
    def get(self, request): return DRFResponse({"message": "FastAPI handled"})

class FacebookCallbackView(APIView):
    def get(self, request): return DRFResponse({"message": "FastAPI handled"})

class InstagramConnectView(APIView):
    def get(self, request): return DRFResponse({"message": "FastAPI handled"})

class InstagramCallbackView(APIView):
    def get(self, request): return DRFResponse({"message": "FastAPI handled"})

