from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User, Company

@receiver(post_save, sender=User)
def create_company(sender, instance, created, **kwargs):
    if created:
        Company.objects.get_or_create(owner=instance)
