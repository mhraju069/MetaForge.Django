from django.shortcuts import render
from core.utils import encrypt_token, decrypt_token
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import SocialAccount
from accounts.helper import get_company
from .serializers import SocialAccountSerializer
# Create your views here.

class SocialAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        pass

    def get(self, request):
        company = get_company(request.user)
        social_accounts = SocialAccount.objects.filter(company=company)
        return Response(SocialAccountSerializer(social_accounts, many=True).data, status=status.HTTP_200_OK)
        
    