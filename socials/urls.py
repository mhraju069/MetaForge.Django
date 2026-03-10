from django.urls import path
from .views import *

# These patterns are now handled by FastAPI in asgi.py
# We keep them here for back-reference and to maintain namespace if needed.
urlpatterns = [
    path('account/', SocialAccountView.as_view(), name='social_account'),
    path('connect/fb/', FacebookConnectView.as_view(), name='facebook_connect'),
    path('callback/fb/', FacebookCallbackView.as_view(), name='facebook_callback'),
    path('connect/ig/', InstagramConnectView.as_view(), name='instagram_connect'),
    path('callback/ig/', InstagramCallbackView.as_view(), name='instagram_callback'),
]