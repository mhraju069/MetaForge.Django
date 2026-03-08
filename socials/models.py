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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.post_id} - {self.account.platform}"


class PostMedia(models.Model):
    post = models.ForeignKey(SocialPost,related_name="media",on_delete=models.CASCADE)
    media = models.ImageField(upload_to="posts")

    def __str__(self):
        return f"{self.post.post_id} - {self.post.page.account.platform}"


