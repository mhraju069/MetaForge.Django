from django.urls import path
from .views import SocialAccountView

urlpatterns = [
    path('account/', SocialAccountView.as_view(), name='social_account'),
]