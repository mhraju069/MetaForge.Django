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
    

class SocialPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialPage
        fields = "__all__"
        read_only_fields = ["id", "account", "created_at", "updated_at"]
    
    def validate(self, attrs):
        account = SocialAccount.objects.get(id=self.context["request"].data["account"])
        if not account:
            raise serializers.ValidationError({"account": "Account not found"})
        if SocialPage.objects.filter(account=account, page_id=attrs["page_id"]).exists():
            raise serializers.ValidationError({"page_id": "Page already exists"})
        return attrs


class SocialPostSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialPost
        fields = "__all__"
        read_only_fields = ["id", "page", "created_at", "updated_at"]
    
    def validate(self, attrs):
        page = SocialPage.objects.get(id=self.context["request"].data["page"])
        if not page:
            raise serializers.ValidationError({"page": "Page not found"})
        if SocialPost.objects.filter(page=page, post_id=attrs["post_id"]).exists():
            raise serializers.ValidationError({"post_id": "Post already exists"})
        return attrs


class PostMediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = PostMedia
        fields = "__all__"
        read_only_fields = ["id", "post", "created_at", "updated_at"]
    
    def validate(self, attrs):
        post = SocialPost.objects.get(id=self.context["request"].data["post"])
        if not post:
            raise serializers.ValidationError({"post": "Post not found"})
        return attrs
