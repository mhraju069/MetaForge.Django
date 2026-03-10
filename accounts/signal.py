from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User, Company
from .ai_helper import train_company_embedding

@receiver(post_save, sender=User)
def create_company(sender, instance, created, **kwargs):
    if created:
        Company.objects.get_or_create(owner=instance)

@receiver(post_save, sender=Company)
def handle_company_training(sender, instance, created, **kwargs):
    """Automatically train company data when changed."""
    # We always retrain if data changes OR it's new
    train_company_embedding(instance)
