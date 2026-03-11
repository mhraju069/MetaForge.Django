import uuid
from django.db import models
from accounts.models import Company

# Create your models here.

class SocialAccount(models.Model):
    PLATFORM_CHOICES = (("fb", "Facebook"),("ig", "Instagram"),("wa", "Whatsapp"),)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, related_name="socials",on_delete=models.CASCADE)

    name = models.CharField(max_length=200,blank=True,null=True)
    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES)   
    account_id = models.CharField(max_length=200)

    token = models.TextField(verbose_name="Encrypted Token")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.company.owner.email} - {self.platform}"


class SocialPost(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(SocialAccount,related_name="posts",on_delete=models.CASCADE)
    post_id = models.CharField(max_length=200)
    caption = models.TextField()
    vector = models.JSONField(null=True, blank=True)
    # AI-detected: Is this post about a product? None=unchecked, True=product, False=not a product
    is_product = models.BooleanField(null=True, blank=True, default=None)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.post_id} - {self.account.platform}"


class PostMedia(models.Model):
    post = models.ForeignKey(SocialPost,related_name="media",on_delete=models.CASCADE)
    media = models.ImageField(upload_to="posts", null=True, blank=True)
    media_url = models.TextField(null=True, blank=True)
    # Perceptual Hash (pHash) - 64-bit fingerprint of the image for visual similarity search
    image_hash = models.CharField(max_length=64, null=True, blank=True, help_text="pHash fingerprint for visual similarity search")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.post.post_id} - {self.post.account.platform}"


class Conversation(models.Model):
    """
    Represents a chat thread between a platform user (customer)
    and the AI shop assistant for a specific social account.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(SocialAccount, related_name="conversations", on_delete=models.CASCADE)
    # The platform user's ID (e.g. Facebook sender_id)
    sender_id = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        # One conversation per user per account
        unique_together = ("account", "sender_id")
        ordering = ["-updated_at"]

    def __str__(self):
        return f"Conv [{self.account.platform}] {self.sender_id}"


class Message(models.Model):
    """
    A single message in a conversation.
    role: 'user' (customer message) or 'assistant' (AI reply)
    """
    ROLE_CHOICES = (("user", "User"), ("assistant", "Assistant"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, related_name="messages", on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    # Optional: store if media was involved
    media_url = models.TextField(null=True, blank=True)
    media_type = models.CharField(max_length=20, null=True, blank=True)  # image, audio, video
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}"
