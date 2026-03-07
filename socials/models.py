import uuid
from django.db import models
from accounts.models import Company

# Create your models here.

class SocialAccount(models.Model):
    PLATFORM_CHOICES = (("fb", "Facebook"),("ig", "Instagram"),("wa", "Whatsapp"),)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, related_name="socials",on_delete=models.CASCADE)

    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES)   
    account_id = models.CharField(max_length=200)

    token = models.TextField(verbose_name="Encrypted Token")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.company.owner.email} - {self.platform}"