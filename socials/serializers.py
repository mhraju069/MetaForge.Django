from rest_framework import serializers
from .models import SocialAccount
from accounts.helper import get_company
from core.utils import decrypt_token

class SocialAccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialAccount
        fields = ["id", "company", "platform", "account_id", "created_at", "updated_at"]
        read_only_fields = ["id", "company", "created_at", "updated_at"]
    
    def validate(self, attrs):
        company = get_company(self.context["request"].user)
        if not company:
            raise serializers.ValidationError({"company": "Company not found"})
        if SocialAccount.objects.filter(company=company, platform=attrs["platform"]).exists():
            raise serializers.ValidationError({"platform": "Social account already exists"})
        return attrs
    