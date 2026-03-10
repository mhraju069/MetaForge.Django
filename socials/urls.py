from django.urls import path
from .views import *
from .webhook import unified_webhook

urlpatterns = [
    path('account/', SocialAccountView.as_view(), name='social_account'),
    path('connect/fb/', FacebookConnectView.as_view(), name='facebook_connect'),
    path('callback/fb/', FacebookCallbackView.as_view(), name='facebook_callback'),
    path('webhook/<str:platform>/', unified_webhook,name='webhook')
]